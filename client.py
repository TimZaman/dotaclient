from collections import Counter
from pprint import pprint, pformat
import argparse
import asyncio
import io
import logging
import math
import os
import pickle
import time
import traceback
import uuid

from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMEMODE_1V1MID
from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.DotaService_pb2 import Actions
from dotaservice.protos.DotaService_pb2 import Empty
from dotaservice.protos.DotaService_pb2 import GameConfig
from dotaservice.protos.DotaService_pb2 import HostMode
from dotaservice.protos.DotaService_pb2 import ObserveConfig
from dotaservice.protos.DotaService_pb2 import Status
from dotaservice.protos.DotaService_pb2 import TEAM_DIRE, TEAM_RADIANT
from grpclib.client import Channel
import aioamqp
import grpc
import numpy as np
import torch

import pika # remove in favour of aioamqp

from policy import Policy

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

torch.manual_seed(7)

# Static variables
OPPOSITE_TEAM = {TEAM_DIRE: TEAM_RADIANT, TEAM_RADIANT: TEAM_DIRE}

# Variables
ROLLOUT_SIZE = 256

TICKS_PER_OBSERVATION = 15
N_DELAY_ENUMS = 5
HOST_TIMESCALE = 10
N_EPISODES = 10000000
MAX_STEPS = 10000

HOST_MODE = HostMode.Value('HOST_MODE_DEDICATED')

DOTASERVICE_HOST = '127.0.0.1'
DOTASERVICE_PORT = 13337

LEARNING_RATE = 1e-4
eps = np.finfo(np.float32).eps.item()

# RMQ
EXPERIENCE_QUEUE_NAME = 'experience'
MODEL_EXCHANGE_NAME = 'model'


# Derivates.
DELAY_ENUM_TO_STEP = math.floor(TICKS_PER_OBSERVATION / N_DELAY_ENUMS)

xp_to_reach_level = {
    1: 0,
    2: 230,
    3: 600,
    4: 1080,
    5: 1680,
    6: 2300,
    7: 2940,
    8: 3600,
    9: 4280,
    10: 5080,
    11: 5900,
    12: 6740,
    13: 7640,
    14: 8865,
    15: 10115,
    16: 11390,
    17: 12690,
    18: 14015,
    19: 15415,
    20: 16905,
    21: 18405,
    22: 20155,
    23: 22155,
    24: 24405,
    25: 26905
}


def get_total_xp(level, xp_needed_to_level):
    if level == 25:
        return xp_to_reach_level[level]
    xp_required_for_next_level = xp_to_reach_level[level + 1] - xp_to_reach_level[level]
    missing_xp_for_next_level = (xp_required_for_next_level - xp_needed_to_level)
    return xp_to_reach_level[level] + missing_xp_for_next_level


def get_reward(prev_obs, obs, player_id):
    """Get the reward."""
    unit_init = get_unit(prev_obs, player_id=player_id)
    unit = get_unit(obs, player_id=player_id)
    player_init = get_player(prev_obs, player_id=player_id)
    player = get_player(obs, player_id=player_id)

    mid_tower_init = get_mid_tower(prev_obs, team_id=player.team_id)
    mid_tower = get_mid_tower(obs, team_id=player.team_id)

    reward = {}

    # XP Reward
    xp_init = get_total_xp(level=unit_init.level, xp_needed_to_level=unit_init.xp_needed_to_level)
    xp = get_total_xp(level=unit.level, xp_needed_to_level=unit.xp_needed_to_level)
    reward['xp'] = (xp - xp_init) * 0.002  # One creep is around 40 xp.

    # HP and death reward
    if unit_init.is_alive and unit.is_alive:
        hp_rel_init = unit_init.health / unit_init.health_max
        hp_rel = unit.health / unit.health_max
        low_hp_factor = 1. + (1 - hp_rel)**2  # hp_rel=0 -> 2; hp_rel=0.5->1.25; hp_rel=1 -> 1.
        reward['hp'] = (hp_rel - hp_rel_init) * low_hp_factor
    else:
        reward['hp'] = 0

    # Kill and death rewards
    reward['kills'] = (player.kills - player_init.kills) * 1.0
    reward['death'] = (player.deaths - player_init.deaths) * -0.5

    # Last-hit reward
    lh = unit.last_hits - unit_init.last_hits
    reward['lh'] = lh * 0.5

    # Deny reward
    denies = unit.denies - unit_init.denies
    reward['denies'] = denies * 0.2

    # Tower hp reward. Note: towers have 1900 hp.
    reward['tower_hp'] = (mid_tower.health - mid_tower_init.health) / 500.

    return reward

policy = Policy()


async def model_callback(channel, body, envelope, properties):
    # TODO(tzaman): add a future so we can wait for first weights
    version = properties.headers['version']
    logger.info("Received new model: version={}, size={}b".format(version, len(body)))
    policy.load_state_dict(torch.load(io.BytesIO(body)), strict=True)
    policy.weight_version = version
    logger.info('Updated weights to version {}'.format(version))


async def setup_model_cb(host, port):
    logger.info('setup_model_cb(host={}, port={})'.format(host, port))
    try:
        transport, protocol = await aioamqp.connect(host=host, port=port, heartbeat=300)
    except aioamqp.AmqpClosedConnection:
        logger.info("closed rmq connections")
        return
    channel = await protocol.channel()
    await channel.exchange(exchange_name=MODEL_EXCHANGE_NAME, type_name='x-recent-history')
    result = await channel.queue(queue_name='', exclusive=True)
    queue_name = result['queue']
    await channel.queue_bind(exchange_name=MODEL_EXCHANGE_NAME, queue_name=queue_name, routing_key='')
    await channel.basic_consume(model_callback, queue_name=queue_name, no_ack=True)


def get_player(state, player_id):
    for player in state.players:
        if player.player_id == player_id:
            return player
    raise ValueError("hero {} not found in state:\n{}".format(player_id, state))


def get_unit(state, player_id):
    for unit in state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') \
            and unit.player_id == player_id:
            return unit
    raise ValueError("unit {} not found in state:\n{}".format(player_id, state))


def get_mid_tower(state, team_id):
    for unit in state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER') \
            and unit.team_id == team_id \
            and 'tower1_mid' in unit.name:
            return unit
    raise ValueError("tower not found in state:\n{}".format(state))


class Player:
    def __init__(self, game_id, player_id, team_id, experience_channel):
        self.player_id = player_id
        self.policy_inputs = []
        self.action_dicts = []
        self.rewards = []
        self.hidden = None
        self.game_id = game_id
        self.team_id = team_id
        self.experience_channel = experience_channel

    def print_reward_summary(self):
        reward_counter = Counter()
        for r in self.rewards:
            reward_counter.update(r)
        reward_counter = dict(reward_counter)

        reward_sum = sum(reward_counter.values())
        logger.info('Player {} reward sum: {:.2f} subrewards:\n{}'.format(
            self.player_id, reward_sum, pformat(reward_counter)))

    def _send_experience_rmq(self):
        logger.debug('_send_experience_rmq')
        data = pickle.dumps({
            'game_id': self.game_id,
            'team_id': self.team_id,
            'player_id': self.player_id,
            'states': self.policy_inputs,
            'actions': self.action_dicts,
            'rewards': self.rewards,
            'weight_version': policy.weight_version,
        })
        self.experience_channel.basic_publish(exchange='', routing_key=EXPERIENCE_QUEUE_NAME, body=data,
            # I don't think we need to make messages persistent (saved to disk)
            # properties=pika.BasicProperties(
            #     delivery_mode = 2, # make message persistent
            # )
        )


    async def rollout(self):
        logger.info('Player {} rollout.'.format(self.player_id))

        if not self.rewards:
            logger.info('nothing to roll out.')
            return

        self.print_reward_summary()

        self._send_experience_rmq()

        # Reset states.
        self.policy_inputs = []
        self.action_dicts = []
        self.rewards = []

    @staticmethod
    def unit_matrix(state, hero_unit, team_id, unit_types, only_self=False):
        handles = []
        m = []
        for unit in state.units:
            if unit.team_id == team_id and unit.is_alive and unit.unit_type in unit_types:
                if only_self:
                    if unit != hero_unit:
                        continue
                rel_hp = (unit.health / unit.health_max) - 0.5
                loc_x = unit.location.x / 7000.
                loc_y = unit.location.y / 7000.
                loc_z = unit.location.z / 7000.
                distance_x = (hero_unit.location.x - unit.location.x)
                distance_y = (hero_unit.location.y - unit.location.y)
                distance = math.sqrt(distance_x**2 + distance_y**2)
                facing_sin = math.sin(unit.facing * (2 * math.pi) / 360)
                facing_cos = math.cos(unit.facing * (2 * math.pi) / 360)
                targettable = float(distance <= hero_unit.attack_range) - 0.5
                distance_x = distance_x / 3000.
                distance_y = distance_y / 3000.
                m.append(
                    torch.Tensor([
                        rel_hp, loc_x, loc_y, loc_z, distance, facing_sin, facing_cos, targettable
                    ]))
                handles.append(unit.handle)
        return m, handles

    def select_action(self, world_state, hidden):
        actions = {}

        # Preprocess the state
        hero_unit = get_unit(world_state, player_id=self.player_id)

        dota_time_norm = world_state.dota_time / 1200.  # Normalize by 20 minutes
        creepwave_sin = math.sin(world_state.dota_time * (2. * math.pi) / 60)

        env_state = torch.Tensor([dota_time_norm, creepwave_sin]).float().unsqueeze(0)

        # Process units
        allied_heroes, allied_hero_handles = self.unit_matrix(
            state=world_state,
            hero_unit=hero_unit,
            unit_types=[CMsgBotWorldState.UnitType.Value('HERO')],
            team_id=hero_unit.team_id,
            only_self=True,  # For now, ignore teammates.
        )

        enemy_heroes, enemy_hero_handles = self.unit_matrix(
            state=world_state,
            hero_unit=hero_unit,
            unit_types=[CMsgBotWorldState.UnitType.Value('HERO')],
            team_id=OPPOSITE_TEAM[hero_unit.team_id],
        )

        allied_nonheroes, allied_nonhero_handles = self.unit_matrix(
            state=world_state,
            hero_unit=hero_unit,
            unit_types=[
                CMsgBotWorldState.UnitType.Value('LANE_CREEP'),
                CMsgBotWorldState.UnitType.Value('CREEP_HERO')
            ],
            team_id=hero_unit.team_id,
        )

        enemy_nonheroes, enemy_nonhero_handles = self.unit_matrix(
            state=world_state,
            hero_unit=hero_unit,
            unit_types=[
                CMsgBotWorldState.UnitType.Value('LANE_CREEP'),
                CMsgBotWorldState.UnitType.Value('CREEP_HERO'),
                CMsgBotWorldState.UnitType.Value('HERO')
            ],
            team_id=OPPOSITE_TEAM[hero_unit.team_id],
        )

        unit_handles = allied_hero_handles + enemy_hero_handles + allied_nonhero_handles + enemy_nonhero_handles

        policy_input = dict(
            env=env_state,
            allied_heroes=allied_heroes,
            enemy_heroes=enemy_heroes,
            allied_nonheroes=allied_nonheroes,
            enemy_nonheroes=enemy_nonheroes,
        )

        head_prob_dict, hidden = policy(**policy_input, hidden=hidden)

        action_dict = policy.select_actions(head_prob_dict=head_prob_dict)

        return action_dict, policy_input, unit_handles, hidden

    def action_to_pb(self, action_dict, state, unit_handles):
        # TODO(tzaman): Recrease the scope of this function. Make it a converter only.
        hero_unit = get_unit(state, player_id=self.player_id)

        action_pb = CMsgBotWorldState.Action()
        # action_pb.actionDelay = action_dict['delay'] * DELAY_ENUM_TO_STEP
        action_enum = action_dict['enum']
        if action_enum == 0:
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
        elif action_enum == 1:
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value(
                'DOTA_UNIT_ORDER_MOVE_TO_POSITION')
            m = CMsgBotWorldState.Action.MoveToLocation()
            hero_location = hero_unit.location
            m.location.x = hero_location.x + Policy.MOVE_ENUMS[action_dict['x']]
            m.location.y = hero_location.y + Policy.MOVE_ENUMS[action_dict['y']]
            m.location.z = 0
            action_pb.moveToLocation.CopyFrom(m)
        elif action_enum == 2:
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value(
                'DOTA_UNIT_ORDER_ATTACK_TARGET')
            m = CMsgBotWorldState.Action.AttackTarget()
            m.target = unit_handles[action_dict['target_unit']]
            m.once = True
            action_pb.attackTarget.CopyFrom(m)
        else:
            raise ValueError("unknown action {}".format(action_enum))
        return action_pb

    def obs_to_action(self, obs):
        action_dict, policy_input, unit_handles, self.hidden = self.select_action(
            world_state=obs,
            hidden=self.hidden,
        )

        self.policy_inputs.append(policy_input)
        self.action_dicts.append(action_dict)

        logger.debug('action:\n' + pformat(action_dict))

        action_pb = self.action_to_pb(action_dict=action_dict, state=obs, unit_handles=unit_handles)
        action_pb.player = self.player_id
        return action_pb

    def compute_reward(self, prev_obs, obs):
        reward = get_reward(prev_obs=prev_obs, obs=obs, player_id=self.player_id)
        self.rewards.append(reward)


class Game:

    ENV_RETRY_DELAY = 15
    EXCEPTION_RETRIES = 10

    def __init__(self, config, dota_service, experience_channel):
        self.config = config
        self.dota_service = dota_service
        self.experience_channel = experience_channel
        self.game_id = 'my_game_id'

    async def play(self):
        logger.info('Starting game.')

        players = {
            TEAM_RADIANT:
            Player(
                game_id=self.game_id,
                player_id=0,
                team_id=TEAM_RADIANT,
                experience_channel=self.experience_channel,
                ),
            TEAM_DIRE:
            Player(
                game_id=self.game_id,
                player_id=5,
                team_id=TEAM_DIRE,
                experience_channel=self.experience_channel,
                ),
        }

        response = await asyncio.wait_for(self.dota_service.reset(self.config), timeout=120)

        prev_obs = {
            TEAM_RADIANT: response.world_state_radiant,
            TEAM_DIRE: response.world_state_dire,
        }
        done = False
        for step in range(MAX_STEPS):  # Steps/actions in the environment
            for team_id, player in players.items():
                logger.debug('step={}, team={}'.format(step, team_id))
                player = players[team_id]

                response = await self.dota_service.observe(ObserveConfig(team_id=team_id))
                if response.status != Status.Value('OK'):
                    done = True
                    break
                obs = response.world_state

                player.compute_reward(prev_obs=prev_obs[team_id], obs=obs)

                action_pb = player.obs_to_action(obs=obs)
                actions_pb = CMsgBotWorldState.Actions(actions=[action_pb])
                actions_pb.dota_time = obs.dota_time

                _ = await self.dota_service.act(Actions(actions=actions_pb, team_id=team_id))

                prev_obs[team_id] = obs

            # Subtract eachothers rewards
            if step > 0:
                rad_rew = sum(players[TEAM_RADIANT].rewards[-1].values())
                dire_rew = sum(players[TEAM_DIRE].rewards[-1].values())
                players[TEAM_RADIANT].rewards[-1]['enemy'] = -dire_rew
                players[TEAM_DIRE].rewards[-1]['enemy'] = -rad_rew

            if step % ROLLOUT_SIZE == 0 and step > 0:
                logger.info('Rollout!')

                for player in players.values():
                    await player.rollout()

            if done:
                break

        # Final rollout. Probably partial.
        for player in players.values():
            await player.rollout()

        # TODO(tzaman): the worldstate ends when game is over. the worldstate doesn't have info
        # about who won the game: so we need to get info from that somehow

        logger.info('Game finished.')


async def main(rmq_host, rmq_port):
    print('main(rmq_host={}, rmq_port={})'.format(rmq_host, rmq_port))
    # RMQ
    rmq_connection = pika.BlockingConnection(pika.ConnectionParameters(host=rmq_host, port=rmq_port, heartbeat=300))
    experience_channel = rmq_connection.channel()
    experience_channel.queue_declare(queue=EXPERIENCE_QUEUE_NAME)

    await setup_model_cb(host=rmq_host, port=rmq_port)

    # Connect to dota
    channel_dota = Channel(DOTASERVICE_HOST, DOTASERVICE_PORT, loop=asyncio.get_event_loop())
    dota_service = DotaServiceStub(channel_dota)

    config = GameConfig(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
        game_mode=DOTA_GAMEMODE_1V1MID,
    )

    game = Game(config=config, dota_service=dota_service, experience_channel=experience_channel)

    for episode in range(0, N_EPISODES):
        logger.info('=== Starting Episode {}.'.format(episode))
        await game.play()

    channel_dota.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", type=str, help="mq ip", default='127.0.0.1')
    parser.add_argument("--port", type=str, help="mq port", default=5672)
    args = parser.parse_args()

    asyncio.run(main(rmq_host=args.ip, rmq_port=args.port))
