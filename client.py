from collections import Counter
from pprint import pprint, pformat
import asyncio
import logging
import math
import os
import pickle
import time
import traceback
import uuid

from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.DotaService_pb2 import Actions
from dotaservice.protos.DotaService_pb2 import Config
from dotaservice.protos.DotaService_pb2 import Empty
from dotaservice.protos.DotaService_pb2 import HostMode
from dotaservice.protos.DotaService_pb2 import Status
from grpclib.client import Channel
from torch.distributions import Categorical
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from protos.DotaOptimizer_pb2 import RolloutData
from protos.DotaOptimizer_pb2 import Empty2
from protos.DotaOptimizer_grpc import DotaOptimizerStub
from policy import Policy

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

torch.manual_seed(7)


# Static variables
TEAM_ID_RADIANT = 2
TEAM_ID_DIRE = 3
OPPOSITE_TEAM = {TEAM_ID_DIRE: TEAM_ID_RADIANT, TEAM_ID_RADIANT: TEAM_ID_DIRE}

# Variables
N_STEPS = 300

TICKS_PER_OBSERVATION = 15
N_DELAY_ENUMS = 5
HOST_TIMESCALE = 10
N_EPISODES = 1000000

HOST_MODE = HostMode.Value('DEDICATED')

OPTIMIZER_HOST = '127.0.0.1'
OPTIMIZER_PORT = 50051

DOTASERVICE_HOST = '127.0.0.1'
DOTASERVICE_PORT = 13337

LEARNING_RATE = 1e-4
eps = np.finfo(np.float32).eps.item()

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
    unit_init = get_hero_unit(prev_obs, player_id=player_id)
    unit = get_hero_unit(obs, player_id=player_id)
    reward = {'xp': 0, 'hp': 0, 'death': 0, 'lh': 0, 'denies': 0}

    # XP Reward
    xp_init = get_total_xp(level=unit_init.level, xp_needed_to_level=unit_init.xp_needed_to_level)
    xp = get_total_xp(level=unit.level, xp_needed_to_level=unit.xp_needed_to_level)
    reward['xp'] = (xp - xp_init) * 0.002  # One creep is around 40 xp.

    # HP and death reward
    if unit_init.is_alive and unit.is_alive:
        hp_rel_init = unit_init.health / unit_init.health_max
        hp_rel = unit.health / unit.health_max
        low_hp_factor = 1. + (1 - hp_rel) ** 2  # hp_rel=0 -> 2; hp_rel=0.5->1.25; hp_rel=1 -> 1.
        reward['hp'] = (hp_rel - hp_rel_init) * low_hp_factor
    if unit_init.is_alive and not unit.is_alive:
        reward['death'] = - 0.5  # Death should be a big penalty

    # Last-hit reward
    lh = unit.last_hits - unit_init.last_hits
    reward['lh'] = lh * 0.5

    # Deny reward
    denies = unit.denies - unit_init.denies
    reward['denies'] = denies * 0.2

    return reward


policy = Policy()
optimizer = optim.SGD(policy.parameters(), lr=LEARNING_RATE)



def get_hero_unit(state, player_id):
    for unit in state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') \
            and unit.player_id == player_id:
            return unit
    raise ValueError("hero {} not found in state:\n{}".format(player_id, state))


class Actor:

    ENV_RETRY_DELAY = 5
    EXCEPTION_RETRIES = 5

    def __init__(self, config, host=DOTASERVICE_HOST, port=DOTASERVICE_PORT, name=''):
        self.host = host
        self.port = port
        self.config = config
        self.name = name
        self.log_prefix = 'Actor {}: '.format(self.name)
        self.env = None
        self.channel = None

    def connect(self):
        if self.channel is None:  # TODO(tzaman) OR channel is closed? How?
            # Set up a channel.
            self.channel = Channel(self.host, self.port, loop=asyncio.get_event_loop())
            self.env = DotaServiceStub(self.channel)
            logger.info(self.log_prefix + 'Channel opened.')

    def disconnect(self):
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self.env = None
            logger.info(self.log_prefix + 'Channel closed.')

    async def __call__(self):
        # When an actor is being called it should first open up a channel. When a channel is opened
        # it makes sense to try to re-use it for this actor. So if a channel has already been
        # opened we should try to reuse.

        for i in range(self.EXCEPTION_RETRIES):
            try:
                while True:
                    self.connect()

                    # Wait for game to boot.
                    response = await asyncio.wait_for(self.env.reset(self.config), timeout=90)
                    initial_obs = response.world_state

                    if response.status == Status.Value('OK'):
                        break
                    else:
                        # Busy channel. Disconnect current and retry.
                        self.disconnect()
                        logger.info(self.log_prefix + "Service not ready, retrying in {}s.".format(
                            self.ENV_RETRY_DELAY))
                        await asyncio.sleep(self.ENV_RETRY_DELAY)

                return await self.call(obs=initial_obs)

            except Exception as e:
                logger.error(self.log_prefix + 'Exception call; retrying ({}/{}).:\n{}'.format(
                    i, self.EXCEPTION_RETRIES, e))
                traceback.print_exc()
                # We always disconnect the channel upon exceptions.
                self.disconnect()
            await asyncio.sleep(1)

    async def call(self, obs):
        logger.info(self.log_prefix + 'Starting game.')
        rewards = []
        log_probs = []
        hidden = None
        player_id = 0


        policy_inputs = []
        action_dicts = []


        for step in range(N_STEPS):  # Steps/actions in the environment
            prev_obs = obs

            action_dict, policy_input, unit_handles, hidden = self.select_action(
                world_state=obs,
                player_id=player_id,
                hidden=hidden,
                )

            policy_inputs.append(policy_input)
            action_dicts.append(action_dict)

            logger.debug('action:\n' + pformat(action_dict))

            # log_probs.append({k: v['logprob'] for k, v in action.items() if 'logprob' in v})

            action_pb = self.action_to_pb(action_dict=action_dict, state=obs, player_id=player_id, unit_handles=unit_handles)
            action_pb.player = player_id

            actions_pb = CMsgBotWorldState.Actions(actions=[action_pb])
            actions_pb.dota_time = obs.dota_time

            response = await asyncio.wait_for(self.env.step(Actions(actions=actions_pb)), timeout=15)
            if response.status != Status.Value('OK'):
                raise ValueError(self.log_prefix + 'Step reponse invalid:\n{}'.format(response))
            obs = response.world_state

            reward = get_reward(prev_obs=prev_obs, obs=obs, player_id=player_id)

            logger.debug(self.log_prefix + 'step={} reward={:.3f}\n'.format(step, sum(reward.values())))
            rewards.append(reward)

        await asyncio.wait_for(self.env.clear(Empty()), timeout=15)

        # Ship the experience.
        channel = Channel(OPTIMIZER_HOST, OPTIMIZER_PORT, loop=asyncio.get_event_loop())
        env = DotaOptimizerStub(channel)
        _ = await env.Rollout(RolloutData(
            game_id='my_game_id',
            actions=pickle.dumps(action_dicts),
            states=pickle.dumps(policy_inputs),
            rewards=pickle.dumps(rewards),
            ))
        channel.close()


        reward_sum = sum([sum(r.values()) for r in rewards])
        logger.info(self.log_prefix + 'Finished. reward_sum={:.2f}'.format(reward_sum))
        return rewards

    def unit_matrix(self, state, hero_unit, team_id, unit_types):
        handles = []
        m = []
        for unit in state.units:
            if unit.team_id == team_id and unit.is_alive and unit.unit_type in unit_types:
                rel_hp = (unit.health / unit.health_max)  # [0 (dead) : 1 (full hp)]
                distance_x = (hero_unit.location.x - unit.location.x) / 3000.
                distance_y = (hero_unit.location.y - unit.location.y) / 3000.
                m.append(torch.Tensor([rel_hp, distance_x, distance_y]))
                handles.append(unit.handle)
        return m, handles

    def select_action(self, world_state, player_id, hidden):
        actions = {}

        # Preprocess the state
        hero_unit = get_hero_unit(world_state, player_id=player_id)

        # Location Input
        location_state = torch.Tensor([hero_unit.location.x, hero_unit.location.y]).unsqueeze(0) / 7000.  # maps the map between [-1 and 1]

        # Health and dotatime input
        hp_rel = 1. - (hero_unit.health / hero_unit.health_max) # Map between [0 and 1]
        dota_time_norm = dota_time = world_state.dota_time / 1200.  # Normalize by 20 minutes
        env_state = torch.Tensor([hp_rel, dota_time_norm]).float().unsqueeze(0) 

        # Process units
        enemy_nonheroes, enemy_nonhero_handles = self.unit_matrix(
            state=world_state,
            hero_unit=hero_unit,
            unit_types=[CMsgBotWorldState.UnitType.Value('LANE_CREEP'), CMsgBotWorldState.UnitType.Value('CREEP_HERO')],
            team_id=OPPOSITE_TEAM[hero_unit.team_id],
            )

        allied_nonheroes, allied_nonhero_handles = self.unit_matrix(
            state=world_state,
            hero_unit=hero_unit,
            unit_types=[CMsgBotWorldState.UnitType.Value('LANE_CREEP'), CMsgBotWorldState.UnitType.Value('CREEP_HERO')],
            team_id=hero_unit.team_id,
            )

        unit_handles = enemy_nonhero_handles + allied_nonhero_handles

        policy_input = dict(
            loc=location_state,
            env=env_state,
            enemy_nonheroes=enemy_nonheroes,
            allied_nonheroes=allied_nonheroes,
        )

        head_prob_dict, hidden = policy(**policy_input, hidden=hidden)

        action_dict = policy.select_actions(head_prob_dict=head_prob_dict)

        return action_dict, policy_input, unit_handles, hidden

    @staticmethod
    def action_to_pb(action_dict, state, player_id, unit_handles):
        # TODO(tzaman): Recrease the scope of this function. Make it a converter only.
        hero_unit = get_hero_unit(state, player_id=player_id)

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



async def main():
    config = Config(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
    )

    actor = Actor(config=config, host=DOTASERVICE_HOST, name=0)

    for episode in range(0, N_EPISODES):
        logger.info('=== Starting Episode {}.'.format(episode))

        # Get the latest weights
        while True:
            try:
                channel = Channel(OPTIMIZER_HOST, OPTIMIZER_PORT, loop=asyncio.get_event_loop())
                env = DotaOptimizerStub(channel)
                response =  await env.GetWeights(Empty2())
                state_dict_p = response.data
                state_dict = pickle.loads(state_dict_p)
                policy.load_state_dict(state_dict, strict=True)
                channel.close()
                break
            except:
                print('Retrying conn..')
                await asyncio.sleep(5)


        all_rewards = []
        start_time = time.time()

        rewards = await actor()
    
        all_rewards.append(rewards)


        time_per_batch = time.time() - start_time
        steps_per_s = len(rewards) / time_per_batch


        reward_counter = Counter()
        for b in all_rewards: # Jobs in a batch.
            for s in b: # Steps in a batch.
                reward_counter.update(s)
        reward_counter = dict(reward_counter)
                
        reward_sum = sum(reward_counter.values())
        avg_reward = reward_sum
        logger.info('Episode={} avg_reward={:.2f} steps/s={:.2f}'.format(
            episode, avg_reward, steps_per_s))
        logger.info('Subrewards:\n{}'.format(pformat(reward_counter)))


if __name__ == '__main__':
    asyncio.run(main())
