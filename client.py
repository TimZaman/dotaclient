from collections import Counter
from pprint import pprint, pformat
import argparse
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
from dotaservice.protos.DotaService_pb2 import Init
from dotaservice.protos.DotaService_pb2 import Team
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
OPPOSITE_TEAM = {Team.Value('DIRE'): Team.Value('RADIANT'), Team.Value('RADIANT'): Team.Value('DIRE')}

# Variables
N_STEPS = 350

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
    unit_init = get_unit(prev_obs, player_id=player_id)
    unit = get_unit(obs, player_id=player_id)
    player_init = get_player(prev_obs, player_id=player_id)
    player = get_player(obs, player_id=player_id)

    reward = {}

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
    else:
        reward['hp'] = 0

    # Kill and death rewards
    reward['kills'] = (player.kills - player_init.kills) * 0.5
    reward['death'] = (player.deaths - player_init.deaths) * - 0.5

    # Last-hit reward
    lh = unit.last_hits - unit_init.last_hits
    reward['lh'] = lh * 0.5

    # Deny reward
    denies = unit.denies - unit_init.denies
    reward['denies'] = denies * 0.2

    return reward


policy = Policy()
optimizer = optim.SGD(policy.parameters(), lr=LEARNING_RATE)


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
    raise ValueError("hero {} not found in state:\n{}".format(player_id, state))


class Player:

    def __init__(self, player_id, team_id):
        self.player_id = player_id
        self.policy_inputs = []
        self.action_dicts = []
        self.rewards = []
        self.hidden = None
        self.team_id = team_id
        self.start_time = time.time()


    @staticmethod
    def unit_matrix(state, hero_unit, team_id, unit_types):
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

    def select_action(self, world_state, hidden):
        actions = {}

        # Preprocess the state
        hero_unit = get_unit(world_state, player_id=self.player_id)

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
            unit_types=[CMsgBotWorldState.UnitType.Value('LANE_CREEP'), CMsgBotWorldState.UnitType.Value('CREEP_HERO'), CMsgBotWorldState.UnitType.Value('HERO')],
            team_id=OPPOSITE_TEAM[hero_unit.team_id],
            )

        allied_nonheroes, allied_nonhero_handles = self.unit_matrix(
            state=world_state,
            hero_unit=hero_unit,
            unit_types=[CMsgBotWorldState.UnitType.Value('LANE_CREEP'), CMsgBotWorldState.UnitType.Value('CREEP_HERO')],
            team_id=hero_unit.team_id,
            )

        enemy_heroes, enemy_hero_handles = self.unit_matrix(
            state=world_state,
            hero_unit=hero_unit,
            unit_types=[CMsgBotWorldState.UnitType.Value('HERO')],
            team_id=hero_unit.team_id,
        )


        unit_handles = enemy_nonhero_handles + allied_nonhero_handles + enemy_hero_handles

        policy_input = dict(
            loc=location_state,
            env=env_state,
            enemy_nonheroes=enemy_nonheroes,
            allied_nonheroes=allied_nonheroes,
            enemy_heroes=enemy_heroes,
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

    def print_summary(self):
        time_per_batch = time.time() - self.start_time
        steps_per_s = len(self.rewards) / time_per_batch

        reward_counter = Counter()
        for r in self.rewards:
            reward_counter.update(r)
        reward_counter = dict(reward_counter)
                
        reward_sum = sum(reward_counter.values())
        logger.info('Player {} reward sum: {:.2f} subrewards:\n{}'.format(
            self.player_id, reward_sum, pformat(reward_counter)))

    async def rollout(self):
        logger.info('::Player:rollout()')

        self.print_summary()

        # Ship the experience.
        channel = Channel(OPTIMIZER_HOST, OPTIMIZER_PORT, loop=asyncio.get_event_loop())
        env = DotaOptimizerStub(channel)
        _ = await env.Rollout(RolloutData(
            game_id='my_game_id',
            actions=pickle.dumps(self.action_dicts),
            states=pickle.dumps(self.policy_inputs),
            rewards=pickle.dumps(self.rewards),
            ))
        channel.close()

    async def go(self, env):
        logger.info('::Player:go()')
        response = await env.initialize(Init(team_id=self.team_id))
        obs = response.world_state

        for step in range(N_STEPS):  # Steps/actions in the environment
            # logger.info('team={} step={} time={}'.format(self.team_id, step, obs.dota_time))
            prev_obs = obs

            action_pb = self.obs_to_action(obs=obs)
            actions_pb = CMsgBotWorldState.Actions(actions=[action_pb])
            actions_pb.dota_time = obs.dota_time

            response = await env.step(Actions(actions=actions_pb, team_id=self.team_id))

            if response.status != Status.Value('OK'):
                raise ValueError(self.log_prefix + 'Step response invalid:\n{}'.format(response))

            obs = response.world_state

            self.compute_reward(prev_obs=prev_obs, obs=obs)

        await env.close(Init(team_id=self.team_id))


class Actor:

    ENV_RETRY_DELAY = 15
    EXCEPTION_RETRIES = 10

    def __init__(self, config, host=DOTASERVICE_HOST, port=DOTASERVICE_PORT):
        self.host = host
        self.port = port
        self.config = config
        # self.log_prefix = 'Actor {}: '.format(self.name)
        self.env = None
        self.channel = None

    def connect(self):
        if self.channel is None:  # TODO(tzaman) OR channel is closed? How?
            # Set up a channel.
            self.channel = Channel(self.host, self.port, loop=asyncio.get_event_loop())
            self.env = DotaServiceStub(self.channel)
            logger.info('Channel opened.')

    def disconnect(self):
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self.env = None
            logger.info('Channel closed.')

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
                    if response.status == Status.Value('OK'):
                        # initial_obs_radiant = response.world_state_radiant
                        # initial_obs_dire = response.world_state_dire
                        break
                    else:
                        # Busy channel. Disconnect current and retry.
                        self.disconnect()
                        logger.info("Service not ready, retrying in {}s.".format(
                            self.ENV_RETRY_DELAY))
                        await asyncio.sleep(self.ENV_RETRY_DELAY)

                return await self.call()  #obs_radiant=initial_obs_radiant, obs_dire=initial_obs_dire)

            except Exception as e:
                logger.error('Exception call; retrying ({}/{}).:\n{}'.format(
                    i, self.EXCEPTION_RETRIES, e))
                traceback.print_exc()
                # We always disconnect the channel upon exceptions.
                self.disconnect()
            await asyncio.sleep(1)
    

    async def call(self):
        logger.info('Starting game.')

        player_radiant = Player(player_id=0, team_id=Team.Value('RADIANT'))
        player_dire = Player(player_id=5, team_id=Team.Value('DIRE'))

        t1 = player_radiant.go(env=self.env)
        t2 = player_dire.go(env=self.env)

        await asyncio.gather(t1, t2)

        await asyncio.wait_for(self.env.clear(Empty()), timeout=15)

        await player_radiant.rollout()
        await player_dire.rollout()

        logger.info('Game finished.')
        
        


async def main():
    config = Config(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
    )

    actor = Actor(config=config, host=DOTASERVICE_HOST)

    for episode in range(0, N_EPISODES):
        logger.info('=== Starting Episode {}.'.format(episode))

        # Connect to optimizer and get the latest weights
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
                logger.info('Connected to optimizer.')
            except Exception as e:
                logger.error('Retrying connection to optimizer. ({})'.format(e))
                await asyncio.sleep(10)

        await actor()

        logger.info('Episode={}'.format(episode))
        
        


if __name__ == '__main__':
    # global OPTIMIZER_HOST
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", type=str, help="optimizer IP", default=OPTIMIZER_HOST)
    args = parser.parse_args()

    OPTIMIZER_HOST = args.ip

    asyncio.run(main())
