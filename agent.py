from collections import Counter
from collections import deque
from datetime import datetime
from functools import lru_cache
from pprint import pprint, pformat
import argparse
import asyncio
import io
import logging
import math
import os
import pickle
import random
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
from dotaservice.protos.DotaService_pb2 import TEAM_DIRE, TEAM_RADIANT, Hero, HeroPick, HeroControlMode
from dotaservice.protos.DotaService_pb2 import HERO_CONTROL_MODE_IDLE, HERO_CONTROL_MODE_DEFAULT, HERO_CONTROL_MODE_CONTROLLED
from dotaservice.protos.DotaService_pb2 import NPC_DOTA_HERO_NEVERMORE, NPC_DOTA_HERO_SNIPER

from grpclib.client import Channel
from tensorboardX import SummaryWriter
import aioamqp
import grpc
import numpy as np
import png
import torch

import pika # TODO(tzaman): remove in favour of aioamqp

from policy import Policy
from policy import REWARD_KEYS

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

# Static variables
OPPOSITE_TEAM = {TEAM_DIRE: TEAM_RADIANT, TEAM_RADIANT: TEAM_DIRE}

TICKS_PER_OBSERVATION = 15
N_DELAY_ENUMS = 5
HOST_TIMESCALE = 10
N_GAMES = 10000000
MAX_AGE_WEIGHTSTORE = 32
MAP_HALF_WIDTH = 7000.  # Approximate size of the half of the map.

HOST_MODE = HostMode.Value('HOST_MODE_DEDICATED')

DOTASERVICE_HOST = '127.0.0.1'
DOTASERVICE_PORT = 13337

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

writer = None

def events_filename_from_writer(writer):
    return writer.file_writer.event_writer._ev_writer._file_name

@lru_cache()
def gcs_bucket():
    from google.cloud import storage
    client = storage.Client()
    bucket = client.get_bucket('dotaservice')
    return bucket


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

    # TODO(tzaman): make a nice reward container?
    reward = {key: 0. for key in REWARD_KEYS}

    # XP Reward
    xp_init = get_total_xp(level=unit_init.level, xp_needed_to_level=unit_init.xp_needed_to_level)
    xp = get_total_xp(level=unit.level, xp_needed_to_level=unit.xp_needed_to_level)
    reward['xp'] = (xp - xp_init) * 0.001  # One creep is around 40 xp.

    # HP and death reward
    if unit_init.is_alive and unit.is_alive:
        hp_rel_init = unit_init.health / unit_init.health_max
        hp_rel = unit.health / unit.health_max
        low_hp_factor = 1. + (1 - hp_rel)**2  # hp_rel=0 -> 2; hp_rel=0.5->1.25; hp_rel=1 -> 1.
        reward['hp'] = (hp_rel - hp_rel_init) * low_hp_factor * 0.2

    # Kill and death rewards
    reward['kills'] = (player.kills - player_init.kills) * 0.4
    reward['death'] = (player.deaths - player_init.deaths) * -0.4

    # Last-hit reward
    lh = unit.last_hits - unit_init.last_hits
    reward['lh'] = lh * 0.1

    # Deny reward
    denies = unit.denies - unit_init.denies
    reward['denies'] = denies * 0.05

    # Tower hp reward. Note: towers have 1900 hp.
    reward['tower_hp'] = (mid_tower.health - mid_tower_init.health) / 1900.

    return reward


class WeightStore:

    def __init__(self, maxlen):
        self.ready = None  # HACK: Will be set to an event
        self.weights = deque(maxlen=maxlen)

        # The latest policy is used as a pointer to the latest and greatest policy. It is updated
        # even while the agents are playing.
        self.latest_policy = Policy()
        self.latest_policy.eval()

    def add(self, version, state_dict):
        # TODO(tzaman): delete old ones
        self.weights.append( (version, state_dict) )

        # Add to latest policy immediatelly
        self.latest_policy.load_state_dict(state_dict, strict=True)
        self.latest_policy.weight_version = version

    def oldest_weights(self):
        return self.weights[0]

    def latest_weights(self):
        return self.weights[-1]

    def load_from_gcs(self, model):
        bucket = gcs_bucket()
        model_blob = bucket.get_blob(model)
        tmp_model = '/tmp/model.pt'
        model_blob.download_to_filename(tmp_model)
        state_dict = torch.load(tmp_model, map_location=lambda storage, loc: storage)
        self.add(version=-1, state_dict=state_dict)
        self.ready.set()

weight_store = WeightStore(maxlen=MAX_AGE_WEIGHTSTORE)


async def model_callback(channel, body, envelope, properties):
    # TODO(tzaman): add a future so we can wait for first weights
    version = properties.headers['version']
    logger.info("Received new model: version={}, size={}b".format(version, len(body)))
    state_dict = torch.load(io.BytesIO(body), map_location=lambda storage, loc: storage)
    weight_store.add(version=version, state_dict=state_dict)
    weight_store.ready.set()


async def rmq_connection_error_cb(exception):
    logger.error('rmq_connection_error_cb(exception={})'.format(exception))
    exit(1)


async def setup_model_cb(host, port):
    # TODO(tzaman): setup proper reconnection, see https://github.com/Polyconseil/aioamqp/issues/65#issuecomment-301737344
    logger.info('setup_model_cb(host={}, port={})'.format(host, port))
    transport, protocol = await aioamqp.connect(
        host=host, port=port, on_error=rmq_connection_error_cb, heartbeat=300)
    channel = await protocol.channel()
    await channel.exchange(exchange_name=MODEL_EXCHANGE_NAME, type_name='x-recent-history',
                           arguments={'x-recent-history-length': 1})
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


def is_unit_attacking_unit(unit_attacker, unit_target):
    # Check for a direct attack.
    if unit_attacker.attack_target_handle == unit_target.handle:
        return 1.
    # Go over the incoming projectiles from this unit.
    for projectile in unit_target.incoming_tracking_projectiles:
        if projectile.caster_handle == unit_attacker.handle and projectile.is_attack:
            return 1.
    # Otherwise, the unit is not attacking the target, and there are no incoming projectiles.
    return 0.

def is_invulnerable(unit):
    for mod in unit.modifiers:
        if mod.name == "modifier_invulnerable":
            return True
    return False

class Player:

    END_STATUS_TO_TEAM = {
        Status.Value('RADIANT_WIN'): TEAM_RADIANT,
        Status.Value('DIRE_WIN'): TEAM_DIRE,
    }

    def __init__(self, game_id, player_id, team_id, hero, experience_channel, use_latest_weights, drawing,
                 validation):
        self.game_id = game_id
        self.player_id = player_id
        self.team_id = team_id
        self.hero = hero
        self.experience_channel = experience_channel
        self.use_latest_weights= use_latest_weights

        self.policy_inputs = []
        self.vec_actions = []
        self.vec_selected_heads_mask = []
        self.rewards = []
        self.hidden = None
        self.drawing = drawing
        self.validation = validation

        self.creeps_had_spawned = False

        use_synced_weights = use_latest_weights and not self.validation

        if use_synced_weights:
            # This will actually use the latest policy, that is even updated while the agent is playing.
            self.policy = weight_store.latest_policy
        else:  # Use non-synchronized weights
            if self.validation or use_latest_weights:
                # Use the latest weights for validation
                version, state_dict = weight_store.latest_weights()
            else:
                # Use the oldest weights.
                version, state_dict = weight_store.oldest_weights()
            self.policy = Policy()
            self.policy.load_state_dict(state_dict, strict=True)
            self.policy.weight_version = version
            self.policy.eval()  # Set to evaluation mode.

        logger.info('Player {} using weights version {}'.format(
            self.player_id, self.policy.weight_version))

    def summed_subrewards(self):
        reward_counter = Counter()
        for r in self.rewards:
            reward_counter.update(r)
        return dict(reward_counter)

    def print_reward_summary(self):
        subrewards = self.summed_subrewards()
        reward_sum = sum(subrewards.values())
        logger.info('Player {} reward sum: {:.2f} subrewards:\n{}'.format(
            self.player_id, reward_sum, pformat(subrewards)))

    def process_endstate(self, end_state):
        # The end-state adds rewards to the last reward.
        if not self.rewards:
            return
        if end_state in self.END_STATUS_TO_TEAM.keys():
            if self.team_id == self.END_STATUS_TO_TEAM[end_state]:
                self.rewards[-1]['win'] = 1
            else:
                self.rewards[-1]['win'] = -1
            return
        
        # Add a negative win reward, because we did not have a clear winner.
        self.rewards[-1]['win'] = -0.25


    @staticmethod
    def pack_policy_inputs(inputs):
        """Convert the list-of-dicts into a dict with a single tensor per input for the sequence."""
        d = { key: [] for key in Policy.INPUT_KEYS}
        for inp in inputs:  # go over steps: (list of dicts)
            for k, v in inp.items(): # go over each input in the step (dict)
                d[k].append(v)

        # Pack it up
        for k, v in d.items():
            # Concatenate together all inputs into a single tensor.
            # We formerly padded this instead of stacking, but that presented issues keeping track
            # of the chosen action ids related to units.
            d[k] = torch.stack(v)
        return d

    @staticmethod
    def pack_rewards(inputs):
        """Pack a list or reward dicts into a dense 2D tensor"""
        t = np.zeros([len(inputs), len(REWARD_KEYS)])
        for i, reward in enumerate(inputs):
            for ir, key in enumerate(REWARD_KEYS):
                t[i, ir] = reward[key]
        return t

    def _send_experience_rmq(self):
        logger.debug('_send_experience_rmq')

        # Pack all the policy inputs into dense tensors
        packed_policy_inputs = self.pack_policy_inputs(inputs=self.policy_inputs)
        packed_rewards = self.pack_rewards(inputs=self.rewards)

        actions = torch.stack(self.vec_actions)
        masks = torch.stack(self.vec_selected_heads_mask)

        data = pickle.dumps({
            'game_id': self.game_id,
            'team_id': self.team_id,
            'player_id': self.player_id,
            'states': packed_policy_inputs,
            'actions': actions,
            'masks': masks,
            'rewards': packed_rewards,
            'weight_version': self.policy.weight_version,
            'canvas': self.drawing.canvas,
        })
        self.experience_channel.basic_publish(
            exchange='', routing_key=EXPERIENCE_QUEUE_NAME, body=data)

    @property
    def steps_queued(self):
        return len(self.rewards)

    def write_validation(self):
        it = self.policy.weight_version
        writer.add_image('game/canvas', self.drawing.canvas, it, dataformats='HWC')
        writer.add_scalar('game/steps', self.steps_queued, it)
        subrewards = self.summed_subrewards()
        reward_sum = sum(subrewards.values())
        writer.add_scalar('game/rewards_sum', reward_sum, it)
        for key, reward in subrewards.items():
            writer.add_scalar('game/rewards_{}'.format(key), reward, it)
        # Upload events to GCS
        writer.file_writer.flush()  # Flush before uploading
        events_filename = events_filename_from_writer(writer)
        blob = gcs_bucket().blob(events_filename)
        blob.upload_from_filename(filename=events_filename)

    async def finish(self):
        if self.validation:
            self.write_validation()
        else:
            await self.rollout()

    async def rollout(self):
        logger.info('Player {} rollout.'.format(self.player_id))

        if not self.rewards:
            logger.info('nothing to roll out.')
            return

        self.print_reward_summary()

        self._send_experience_rmq()

        # Reset states.
        self.policy_inputs = []
        self.vec_actions = []
        self.rewards = []
        self.vec_selected_heads_mask = []

    @staticmethod
    def unit_separation(state, team_id):
        # Break apart the full unit-list into specific categories for allied and
        # enemy unit groups of various types so we don't have to repeatedly iterate
        # the full unit-list again.
        allied_heroes       = []
        enemy_heroes        = []
        allied_nonheroes    = []
        enemy_nonheroes     = []
        allied_creep        = []
        enemy_creep         = []
        allied_towers       = []
        enemy_towers        = []
        for unit in state.units:
            # check if allied or enemy unit
            if unit.team_id == team_id:
                if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO'):
                    allied_heroes.append(unit)
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('CREEP_HERO'):
                    allied_nonheroes.append(unit)
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP'):
                    allied_creep.append(unit)
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER'):
                    if unit.name[-5:] == "1_mid":  # Only consider the mid tower for now.
                        allied_towers.append(unit)
            else:
                if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO'):
                    enemy_heroes.append(unit)
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('CREEP_HERO'):
                    enemy_nonheroes.append(unit)
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP'):
                    enemy_creep.append(unit)
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER'):
                    if unit.name[-5:] == "1_mid":  # Only consider the mid tower for now.
                        enemy_towers.append(unit)

        return allied_heroes, enemy_heroes, allied_nonheroes, enemy_nonheroes, \
               allied_creep, enemy_creep, allied_towers, enemy_towers


    @staticmethod
    def unit_matrix(unit_list, hero_unit, only_self=False, max_units=16):
        # We are always inserting an 'zero' unit to make sure the policy doesn't barf
        # We can't just pad this, because we will otherwise lose track of corresponding chosen
        # actions relating to output indices. Even if we would, batching multiple sequences together
        # would then be another error prone nightmare.
        handles = torch.full([max_units], -1)
        m = torch.zeros(max_units, 10)
        i = 0
        for unit in unit_list:
            if unit.is_alive:
                if only_self:
                    if unit != hero_unit:
                        continue
                if i >= max_units:
                    break
                rel_hp = 1.0 - (unit.health / unit.health_max)
                rel_mana = 0.0
                if unit.mana_max > 0:
                    rel_mana = 1.0 - (unit.mana / unit.mana_max)
                loc_x = unit.location.x / MAP_HALF_WIDTH
                loc_y = unit.location.y / MAP_HALF_WIDTH
                loc_z = (unit.location.z / 512.)-0.5
                distance_x = (hero_unit.location.x - unit.location.x)
                distance_y = (hero_unit.location.y - unit.location.y)
                distance = math.sqrt(distance_x**2 + distance_y**2)
                norm_distance = (distance / MAP_HALF_WIDTH) - 0.5

                # Get the direction where the unit is facing.
                facing_sin = math.sin(unit.facing * (2 * math.pi) / 360)	
                facing_cos = math.cos(unit.facing * (2 * math.pi) / 360)

                # Calculates normalized boolean value [-0.5 or 0.5] of if unit is within 
                # attack range of hero.
                in_attack_range = float(distance <= hero_unit.attack_range) - 0.5

                # Calculates normalized boolean value [-0.5 or 0.5] of if that unit
                # is currently targeting me with right-click attacks.
                is_attacking_me = float(is_unit_attacking_unit(unit, hero_unit)) - 0.5
                me_attacking_unit = float(is_unit_attacking_unit(hero_unit, unit)) - 0.5

                m[i] = (
                    # TODO(tzaman): Add rel_mana, norm_distance once it makes sense
                    torch.tensor([
                        rel_hp, loc_x, loc_y, loc_z, norm_distance, facing_sin, facing_cos,
                        in_attack_range, is_attacking_me, me_attacking_unit
                    ]))

                # Because we are currently only attacking, check if these units are valid
                # HACK: Make a nice interface for this, per enum used?
                if unit.is_invulnerable or unit.is_attack_immune:
                    handles[i] = -1
                elif unit.team_id == hero_unit.team_id and unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER'):
                    # Its own tower:
                    handles[i] = -1
                elif unit.team_id == hero_unit.team_id and (unit.health / unit.health_max) > 0.5:
                    # Not denyable
                    handles[i] = -1
                else:
                    handles[i] = unit.handle

                i += 1
        return m, handles

    def select_action(self, world_state):
        # Preprocess the state
        hero_unit = get_unit(world_state, player_id=self.player_id)

        dota_time_norm = world_state.dota_time / 1200.  # Normalize by 20 minutes
        creepwave_sin = math.sin(world_state.dota_time * (2. * math.pi) / 60)
        team_float = -.2 if self.team_id == TEAM_DIRE else .2

        env_state = torch.Tensor([dota_time_norm, creepwave_sin, team_float])

        # Separate units into unit-type groups for both teams
        # The goal is to iterate only once through the entire unit list
        # in the provided world-state protobuf and for further filtering
        # only iterate across the unit-type specific list of interest.
        ah, eh, anh, enh, ac, ec, at, et = self.unit_separation(world_state, hero_unit.team_id)

        # Process units into Tensors & Handles
        allied_heroes, allied_hero_handles = self.unit_matrix(
            unit_list=ah,
            hero_unit=hero_unit,
            only_self=True,  # For now, ignore teammates.
            max_units=1,
        )

        enemy_heroes, enemy_hero_handles = self.unit_matrix(
            unit_list=eh,
            hero_unit=hero_unit,
            max_units=5,
        )

        allied_nonheroes, allied_nonhero_handles = self.unit_matrix(
            unit_list=[*anh, *ac],
            hero_unit=hero_unit,
            max_units=16,
        )

        enemy_nonheroes, enemy_nonhero_handles = self.unit_matrix(
            unit_list=[*enh, *ec],
            hero_unit=hero_unit,
            max_units=16,
        )

        allied_towers, allied_tower_handles = self.unit_matrix(
            unit_list=at,
            hero_unit=hero_unit,
            max_units=1,
        )

        enemy_towers, enemy_tower_handles = self.unit_matrix(
            unit_list=et,
            hero_unit=hero_unit,
            max_units=1,
        )

        unit_handles = torch.cat([allied_hero_handles, enemy_hero_handles, allied_nonhero_handles, enemy_nonhero_handles,
                                  allied_tower_handles, enemy_tower_handles])

        if not self.creeps_had_spawned and world_state.dota_time > 0.:
            # Check that creeps have spawned. See dotaclient/issues/15.
            # TODO(tzaman): this should be handled by DotaService.
            # self.creeps_had_spawned = bool((allied_nonhero_handles != -1).any())
            self.creeps_had_spawned = len(ac) > 0
            if not self.creeps_had_spawned:
                raise ValueError('Creeps have not spawned at timestep {}'.format(world_state.dota_time))

        policy_input = {
            'env': env_state,
            'allied_heroes': allied_heroes,
            'enemy_heroes': enemy_heroes,
            'allied_nonheroes': allied_nonheroes,
            'enemy_nonheroes': enemy_nonheroes,
            'allied_towers': allied_towers,
            'enemy_towers': enemy_towers,
        }

        logger.debug('policy_input:\n' + pformat(policy_input))

        head_logits_dict, value, self.hidden = self.policy.single(**policy_input, hidden=self.hidden)

        logger.debug('head_logits_dict:\n' + pformat(head_logits_dict))
        logger.debug('value={}'.format(value))

        # Select valid actions. This mask contains all viable actions.
        action_masks = Policy.action_masks(unit_handles=unit_handles)
        logger.debug('action_masks:\n' + pformat(action_masks))

        # Perform a masked softmax
        head_prob_dict = {}
        for key in head_logits_dict:
            head_prob_dict[key] = Policy.masked_softmax(x=head_logits_dict[key], mask=action_masks[key])

        logger.debug('head_prob_dict (masked):\n' + pformat(head_prob_dict))

        action_dict = Policy.select_actions(head_prob_dict=head_prob_dict)

        # Given the action selections, get the head mask.
        head_masks = Policy.head_masks(selections=action_dict)
        logger.debug('head_masks:\n' + pformat(head_masks))

        # Combine the head mask and the selection mask, to get all relevant probabilities of the
        # current action.
        selected_heads_mask = {key: head_masks[key] & action_masks[key] for key in head_masks}
        logger.debug('selected_heads_mask:\n' + pformat(selected_heads_mask))

        return policy_input, action_dict, selected_heads_mask, unit_handles

    def action_to_pb(self, action_dict, state, unit_handles):
        # TODO(tzaman): Recrease the scope of this function. Make it a converter only.
        hero_unit = get_unit(state, player_id=self.player_id)

        action_pb = CMsgBotWorldState.Action()
        action_pb.actionDelay = 0  # action_dict['delay'] * DELAY_ENUM_TO_STEP
        action_enum = action_dict['enum']
        if action_enum == 0:
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
        elif action_enum == 1:
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value(
                'DOTA_UNIT_ORDER_MOVE_DIRECTLY')
            m = CMsgBotWorldState.Action.MoveToLocation()
            hero_location = hero_unit.location
            m.location.x = hero_location.x + Policy.MOVE_ENUMS[action_dict['x']]
            m.location.y = hero_location.y + Policy.MOVE_ENUMS[action_dict['y']]
            m.location.z = 0
            action_pb.moveDirectly.CopyFrom(m)
        elif action_enum == 2:
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value(
                'DOTA_UNIT_ORDER_ATTACK_TARGET')
            m = CMsgBotWorldState.Action.AttackTarget()
            if 'target_unit' in action_dict:
                m.target = unit_handles[action_dict['target_unit']]
            else:
                m.target = -1
            m.once = True
            action_pb.attackTarget.CopyFrom(m)
        else:
            raise ValueError("unknown action {}".format(action_enum))
        return action_pb

    def obs_to_action(self, obs):
        policy_input, action_dict, selected_heads_mask, unit_handles = self.select_action(
            world_state=obs,
        )

        self.policy_inputs.append(policy_input)
        self.vec_actions.append(Policy.flatten_selections(action_dict))
        self.vec_selected_heads_mask.append(Policy.flatten_head(inputs=selected_heads_mask).view(-1))
  
        logger.debug('action:\n' + pformat(action_dict))

        action_pb = self.action_to_pb(action_dict=action_dict, state=obs, unit_handles=unit_handles)
        action_pb.player = self.player_id
        return action_pb

    def compute_reward(self, prev_obs, obs):
        # Draw.
        self.drawing.step(state=obs, team_id=self.team_id, player_id=self.player_id)

        reward = get_reward(prev_obs=prev_obs, obs=obs, player_id=self.player_id)
        self.rewards.append(reward)


class Drawing:

    TEAM_COLORS = {TEAM_DIRE: [255, 0, 0], TEAM_RADIANT: [0, 255, 0]}

    def __init__(self, size=256):
        # Notice the shape is in (H, W, C)
        self.size = size
        self.sizeh = self.size / 2.
        self.canvas = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255
        self.ratio = self.sizeh / (8000.)

    def normalize_location(self, l):
        return int((l.x * self.ratio) + self.sizeh), int(self.size - (l.y * self.ratio) - self.sizeh)

    def step(self, state, team_id, player_id):
        for unit in state.units:
            if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') \
                and unit.player_id == player_id:
                x, y = self.normalize_location(l=unit.location)
                self.canvas[y, x] = self.TEAM_COLORS[team_id]

    def save(self, stem):
        png.from_array(self.canvas, 'RGB').save('{}.png'.format(stem))


class Game:

    ENV_RETRY_DELAY = 15

    def __init__(self, dota_service, experience_channel, rollout_size, max_dota_time,
                 latest_weights_prob, validation):
        self.dota_service = dota_service
        self.experience_channel = experience_channel
        self.rollout_size = rollout_size
        self.max_dota_time = max_dota_time
        self.latest_weights_prob = latest_weights_prob
        self.validation = validation

    async def play(self, config, game_id):
        logger.info('Starting game.')

        use_latest_weights = {TEAM_RADIANT: True, TEAM_DIRE: True}
        if random.random() > self.latest_weights_prob:
            old_model_team = random.choice([TEAM_RADIANT, TEAM_DIRE])
            use_latest_weights[old_model_team] = False

        drawing = Drawing()  # TODO(tzaman): drawing should include include what's visible to the player

        # Reset and obtain the initial observation. This dictates who we are controlling,
        # this is done before the player definition, because there might be humand playing
        # that take up bot positions.
        response = await asyncio.wait_for(self.dota_service.reset(config), timeout=120)

        player_request = config.hero_picks
        players_response = response.players  # Lists all human and bot players.
        players = {TEAM_RADIANT: [], TEAM_DIRE: []}
        for p_req, p_res in zip(player_request, players_response):
            assert p_req.team_id == p_req.team_id  # TODO(tzaman): more tests?
            if p_res.is_bot and p_req.control_mode == HERO_CONTROL_MODE_CONTROLLED:
                player = Player(
                    game_id=game_id,
                    player_id=p_res.id,
                    team_id=p_res.team_id,
                    hero=p_res.hero,
                    experience_channel=self.experience_channel,
                    use_latest_weights=use_latest_weights[p_res.team_id],
                    drawing=drawing,
                    validation=self.validation,
                )
                players[p_res.team_id].append(player)

        prev_obs = {
            TEAM_RADIANT: response.world_state_radiant,
            TEAM_DIRE: response.world_state_dire,
        }
        done = False
        step = 0
        dota_time = -float('Inf')
        end_state = None
        while dota_time < self.max_dota_time:
            reward_sum_step = {TEAM_RADIANT:0, TEAM_DIRE:0}
            for team_id in [TEAM_RADIANT, TEAM_DIRE]:
                logger.debug('\ndota_time={:.2f}, team={}'.format(dota_time, team_id))

                response = await self.dota_service.observe(ObserveConfig(team_id=team_id))
                if response.status != Status.Value('OK'):
                    end_state = response.status
                    done = True
                    break
                obs = response.world_state
                dota_time = obs.dota_time

                # We not loop over each player in this team and get each players action.
                actions = []
                for player in players[team_id]:
                    player.compute_reward(prev_obs=prev_obs[team_id], obs=obs)
                    reward_sum_step[team_id] += sum(player.rewards[-1].values())
                    with torch.no_grad():
                        action_pb = player.obs_to_action(obs=obs)
                    actions.append(action_pb)

                actions_pb = CMsgBotWorldState.Actions(actions=actions)
                actions_pb.dota_time = obs.dota_time

                _ = await self.dota_service.act(Actions(actions=actions_pb, team_id=team_id))

                prev_obs[team_id] = obs

            if not self.validation:
                # Subtract eachothers rewards
                for team_id in [TEAM_RADIANT, TEAM_DIRE]:
                    for player in players[team_id]:
                        player.rewards[-1]['enemy'] = -reward_sum_step[OPPOSITE_TEAM[team_id]]

                for player in [*players[TEAM_RADIANT], *players[TEAM_DIRE]]:
                    if player.steps_queued > 0 and player.steps_queued % self.rollout_size == 0:
                        await player.rollout()

            if done:
                break

        # drawing.save(stem=game_id)  # HACK

        # Finish (e.g. final rollout or send validation metrics).
        for player in [*players[TEAM_RADIANT], *players[TEAM_DIRE]]:
            player.process_endstate(end_state)
            await player.finish()

        # TODO(tzaman): the worldstate ends when game is over. the worldstate doesn't have info
        # about who won the game: so we need to get info from that somehow

        logger.info('Game finished.')


async def main(rmq_host, rmq_port, rollout_size, max_dota_time, latest_weights_prob, initial_model, validation,
               log_dir):
    logger.info('main(rmq_host={}, rmq_port={})'.format(rmq_host, rmq_port))

    # RMQ
    rmq_connection = pika.BlockingConnection(pika.ConnectionParameters(host=rmq_host, port=rmq_port, heartbeat=300))
    experience_channel = rmq_connection.channel()
    experience_channel.queue_declare(queue=EXPERIENCE_QUEUE_NAME)

    weight_store.ready = asyncio.Event(loop=asyncio.get_event_loop())

    global writer
    writer = SummaryWriter(log_dir=log_dir)

    # Optionally
    if initial_model:
        weight_store.load_from_gcs(initial_model)

    # Set up the model callback.
    await setup_model_cb(host=rmq_host, port=rmq_port)

    # Wait for the first model weight to come in.
    await weight_store.ready.wait()

    # Connect to dota
    channel_dota = Channel(DOTASERVICE_HOST, DOTASERVICE_PORT, loop=asyncio.get_event_loop())
    dota_service = DotaServiceStub(channel_dota)

    game = Game(dota_service=dota_service, experience_channel=experience_channel,
                rollout_size=rollout_size, max_dota_time=max_dota_time,
                latest_weights_prob=latest_weights_prob, validation=validation)

    if validation:
        config = get_1v1_bot_vs_default_config()
    else:
        config = get_1v1_selfplay_config()

    for i in range(0, N_GAMES):
        logger.info('=== Starting Game {}.'.format(i))
        game_id = str(datetime.now().strftime('%b%d_%H-%M-%S'))

        try:
            await game.play(config=config, game_id=game_id)
        except:
            traceback.print_exc()
            return

    channel_dota.close()


def get_1v1_bot_vs_default_config():
    hero_picks = [
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_NEVERMORE, control_mode=HERO_CONTROL_MODE_DEFAULT),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_NEVERMORE, control_mode=HERO_CONTROL_MODE_CONTROLLED),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
    ]
    return GameConfig(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
        game_mode=DOTA_GAMEMODE_1V1MID,
        hero_picks=hero_picks,
    )


def get_1v1_selfplay_config():
    hero_picks = [
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_NEVERMORE, control_mode=HERO_CONTROL_MODE_CONTROLLED),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_RADIANT, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_NEVERMORE, control_mode=HERO_CONTROL_MODE_CONTROLLED),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
        HeroPick(team_id=TEAM_DIRE, hero_id=NPC_DOTA_HERO_SNIPER, control_mode=HERO_CONTROL_MODE_IDLE),
    ]
    return GameConfig(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
        game_mode=DOTA_GAMEMODE_1V1MID,
        hero_picks=hero_picks,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", type=str, help="mq ip", default='127.0.0.1')
    parser.add_argument("--port", type=int, help="mq port", default=5672)
    parser.add_argument("--rollout-size", type=int, help="size of each rollout (steps)", default=256)
    parser.add_argument("--max-dota-time", type=int, help="Maximum in-game (dota) time of a game before restarting", default=600)
    parser.add_argument("-l", "--log", dest="log_level", help="Set the logging level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument("--model", type=str, help="Initial model to immediatelly start")
    parser.add_argument("--use-latest-weights-prob", type=float,
                        help="Probability of using the latest weights. Otherwise some old one is chosen if available.", default=1.0)
    parser.add_argument("--validation", type=bool, help="Function as validation runner", default=False)
    parser.add_argument("--log-dir", type=str, help="Logging directory", default='')
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    loop = asyncio.get_event_loop()
    coro = main(rmq_host=args.ip, rmq_port=args.port, rollout_size=args.rollout_size,
                max_dota_time=args.max_dota_time, latest_weights_prob=args.use_latest_weights_prob,
                initial_model=args.model, validation=args.validation, log_dir=args.log_dir)
    loop.run_until_complete(coro)
