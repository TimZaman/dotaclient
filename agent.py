from collections import Counter
from collections import deque
from datetime import datetime
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
from dotaservice.protos.DotaService_pb2 import TEAM_DIRE, TEAM_RADIANT
from grpclib.client import Channel
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
    reward['kills'] = (player.kills - player_init.kills) * 2.0
    reward['death'] = (player.deaths - player_init.deaths) * -3.0

    # Last-hit reward
    lh = unit.last_hits - unit_init.last_hits
    reward['lh'] = lh * 0.5

    # Deny reward
    denies = unit.denies - unit_init.denies
    reward['denies'] = denies * 0.2

    # Tower hp reward. Note: towers have 1900 hp.
    reward['tower_hp'] = (mid_tower.health - mid_tower_init.health) / 500.

    return reward


class WeightStore:

    def __init__(self, maxlen):
        self.ready = None  # HACK: Will be set to an event
        self.weights = deque(maxlen=maxlen)

    def add(self, version, state_dict):
        # TODO(tzaman): delete old ones
        self.weights.append( (version, state_dict) )

    def oldest_model(self):
        return self.weights[0]

    def latest_model(self):
        return self.weights[-1]

    def load_from_gcs(self, model):
        from google.cloud import storage
        client = storage.Client()
        bucket = client.get_bucket('dotaservice')
        model_blob = bucket.get_blob(model)
        tmp_model = '/tmp/model.pt'
        model_blob.download_to_filename(tmp_model)
        state_dict = torch.load(tmp_model)
        self.add(version=-1, state_dict=state_dict)
        self.ready.set()

weight_store = WeightStore(maxlen=MAX_AGE_WEIGHTSTORE)


async def model_callback(channel, body, envelope, properties):
    # TODO(tzaman): add a future so we can wait for first weights
    version = properties.headers['version']
    logger.info("Received new model: version={}, size={}b".format(version, len(body)))
    state_dict = torch.load(io.BytesIO(body))
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


class Player:

    END_STATUS_TO_TEAM = {
        Status.Value('RADIANT_WIN'): TEAM_RADIANT,
        Status.Value('DIRE_WIN'): TEAM_DIRE,
    }

    def __init__(self, game_id, player_id, team_id, experience_channel, use_latest_model, drawing):
        self.game_id = game_id
        self.player_id = player_id
        self.team_id = team_id
        self.experience_channel = experience_channel
        self.use_latest_model= use_latest_model

        self.policy_inputs = []
        self.vec_actions = []
        self.vec_selected_heads_mask = []
        self.rewards = []
        self.hidden = None
        self.drawing = drawing

        self.creeps_had_spawned = False

        if use_latest_model:
            version, state_dict = weight_store.latest_model()
        else:
            version, state_dict = weight_store.oldest_model()

        self.policy = Policy()
        self.policy.load_state_dict(state_dict, strict=False)  # TODO(tzaman): HACK: should be true
        self.policy.weight_version = version

        logger.info('Player {} using weights version {}'.format(
            self.player_id, self.policy.weight_version))

    def print_reward_summary(self):
        reward_counter = Counter()
        for r in self.rewards:
            reward_counter.update(r)
        reward_counter = dict(reward_counter)

        reward_sum = sum(reward_counter.values())
        logger.info('Player {} reward sum: {:.2f} subrewards:\n{}'.format(
            self.player_id, reward_sum, pformat(reward_counter)))

    def process_endstate(self, end_state):
        # The end-state adds rewards to the last reward.
        if not self.rewards:
            return
        if end_state in self.END_STATUS_TO_TEAM.keys():
            if self.team_id == self.END_STATUS_TO_TEAM[end_state]:
                self.rewards[-1]['win'] = 5
            else:
                self.rewards[-1]['win'] = -5

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

        data = pickle.dumps({
            'game_id': self.game_id,
            'team_id': self.team_id,
            'player_id': self.player_id,
            'states': packed_policy_inputs,
            'actions': torch.stack(self.vec_actions),
            'masks': torch.stack(self.vec_selected_heads_mask),
            'rewards': packed_rewards,
            'weight_version': self.policy.weight_version,
            'canvas': self.drawing.canvas,
        })
        self.experience_channel.basic_publish(
            exchange='', routing_key=EXPERIENCE_QUEUE_NAME, body=data)

    @property
    def steps_queued(self):
        return len(self.rewards)

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

    @staticmethod
    def unit_matrix(state, hero_unit, team_id, unit_types, only_self=False, max_units=16):
        # We are always inserting an 'zero' unit to make sure the policy doesn't barf
        # We can't just pad this, because we will otherwise lose track of corresponding chosen
        # actions relating to output indices. Even if we would, batching multiple sequences together
        # would then be another error prone nightmare.
        handles = torch.full([max_units], -1)
        m = torch.zeros(max_units, 8)
        i = 0
        for unit in state.units:
            if unit.team_id == team_id and unit.is_alive and unit.unit_type in unit_types:
                if only_self:
                    if unit != hero_unit:
                        continue
                if i >= max_units:
                    break
                rel_hp = 1.0 - (unit.health / unit.health_max)
                rel_mana = 0.0
                if unit.mana_max > 0:
                    rel_mana = 1.0 - (unit.mana / unit.mana_max)
                loc_x = unit.location.x / 7000.
                loc_y = unit.location.y / 7000.
                loc_z = (unit.location.z / 512.)-0.5
                distance_x = (hero_unit.location.x - unit.location.x)
                distance_y = (hero_unit.location.y - unit.location.y)
                distance = math.sqrt(distance_x**2 + distance_y**2)
                facing_sin = math.sin(unit.facing * (2 * math.pi) / 360)
                facing_cos = math.cos(unit.facing * (2 * math.pi) / 360)
                in_attack_range = float(distance <= hero_unit.attack_range) - 0.5
                distance = (distance / 7000.) - 0.5
                # TODO(tzaman): use distance x,y
                # distance_x = (distance_x / 3000)-0.5.
                # distance_y = (distance_y / 3000)-0.5.
                # TODO(tzaman): Add rel mana once it makes sense
                m[i] = (
                    torch.tensor([
                        rel_hp, loc_x, loc_y, loc_z, distance, facing_sin, facing_cos,
                        in_attack_range,
                    ]))
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

        # Process units
        allied_heroes, allied_hero_handles = self.unit_matrix(
            state=world_state,
            hero_unit=hero_unit,
            unit_types=[CMsgBotWorldState.UnitType.Value('HERO')],
            team_id=hero_unit.team_id,
            only_self=True,  # For now, ignore teammates.
            max_units=1,
        )

        enemy_heroes, enemy_hero_handles = self.unit_matrix(
            state=world_state,
            hero_unit=hero_unit,
            unit_types=[CMsgBotWorldState.UnitType.Value('HERO')],
            team_id=OPPOSITE_TEAM[hero_unit.team_id],
            max_units=5,
        )

        allied_nonheroes, allied_nonhero_handles = self.unit_matrix(
            state=world_state,
            hero_unit=hero_unit,
            unit_types=[
                CMsgBotWorldState.UnitType.Value('LANE_CREEP'),
                CMsgBotWorldState.UnitType.Value('CREEP_HERO')
            ],
            team_id=hero_unit.team_id,
            max_units=16,
        )

        enemy_nonheroes, enemy_nonhero_handles = self.unit_matrix(
            state=world_state,
            hero_unit=hero_unit,
            unit_types=[
                CMsgBotWorldState.UnitType.Value('LANE_CREEP'),
                CMsgBotWorldState.UnitType.Value('CREEP_HERO')
            ],
            team_id=OPPOSITE_TEAM[hero_unit.team_id],
            max_units=16,
        )

        unit_handles = torch.cat([allied_hero_handles, enemy_hero_handles, allied_nonhero_handles, enemy_nonhero_handles])

        if not self.creeps_had_spawned and world_state.dota_time > 0.:
            # Check that creeps have spawned. See dotaclient/issues/15.
            # TODO(tzaman): should this be handled by DotaService?
            self.creeps_had_spawned = bool((allied_nonhero_handles != -1).any())
            if not self.creeps_had_spawned:
                raise ValueError('Creeps have not spawned at timestep {}'.format(world_state.dota_time))

        policy_input = dict(
            env=env_state,
            allied_heroes=allied_heroes,
            enemy_heroes=enemy_heroes,
            allied_nonheroes=allied_nonheroes,
            enemy_nonheroes=enemy_nonheroes,
        )

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
        # action_pb.actionDelay = action_dict['delay'] * DELAY_ENUM_TO_STEP
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

    def __init__(self, config, dota_service, experience_channel, rollout_size, max_dota_time,
                 latest_model_prob):
        self.config = config
        self.dota_service = dota_service
        self.experience_channel = experience_channel
        self.rollout_size = rollout_size
        self.max_dota_time = max_dota_time
        self.latest_model_prob = latest_model_prob

    async def play(self, game_id):
        logger.info('Starting game.')

        use_latest_model = {TEAM_RADIANT: True, TEAM_DIRE: True}
        if random.random() > self.latest_model_prob:
            old_model_team = random.choice([TEAM_RADIANT, TEAM_DIRE])
            use_latest_model[old_model_team] = False

        drawing = Drawing()  # TODO(tzaman): drawing should include include what's visible to the player

        players = {
            TEAM_RADIANT:
            Player(
                game_id=game_id,
                player_id=0,
                team_id=TEAM_RADIANT,
                experience_channel=self.experience_channel,
                use_latest_model=use_latest_model[TEAM_RADIANT],
                drawing=drawing,
                ),
            TEAM_DIRE:
            Player(
                game_id=game_id,
                player_id=5,
                team_id=TEAM_DIRE,
                experience_channel=self.experience_channel,
                use_latest_model=use_latest_model[TEAM_DIRE],
                drawing=drawing,
                ),
        }
        
        response = await asyncio.wait_for(self.dota_service.reset(self.config), timeout=120)

        prev_obs = {
            TEAM_RADIANT: response.world_state_radiant,
            TEAM_DIRE: response.world_state_dire,
        }
        done = False
        step = 0
        dota_time = -float('Inf')
        end_state = None
        while dota_time < self.max_dota_time:
            for team_id, player in players.items():  # TODO(tzaman): actually, should loop over teams first instead of players.
                logger.debug('dota_time={:.2f}, team={}'.format(dota_time, team_id))
                player = players[team_id]

                response = await self.dota_service.observe(ObserveConfig(team_id=team_id))
                if response.status != Status.Value('OK'):
                    end_state = response.status
                    done = True
                    break
                obs = response.world_state
                dota_time = obs.dota_time

                player.compute_reward(prev_obs=prev_obs[team_id], obs=obs)

                action_pb = player.obs_to_action(obs=obs)
                actions_pb = CMsgBotWorldState.Actions(actions=[action_pb])
                actions_pb.dota_time = obs.dota_time

                _ = await self.dota_service.act(Actions(actions=actions_pb, team_id=team_id))

                prev_obs[team_id] = obs

            # Subtract eachothers rewards
            # TODO(tzaman): notice that the endstate (win reward) processing is not captured here.
            rad_rew = sum(players[TEAM_RADIANT].rewards[-1].values())
            dire_rew = sum(players[TEAM_DIRE].rewards[-1].values())
            players[TEAM_RADIANT].rewards[-1]['enemy'] = -dire_rew
            players[TEAM_DIRE].rewards[-1]['enemy'] = -rad_rew

            for player in players.values():
                if player.steps_queued > 0 and player.steps_queued % self.rollout_size == 0:
                    await player.rollout()

            if done:
                break

        if end_state is not None:
            
            for player in players.values():
                player.process_endstate(end_state)

        # drawing.save(stem=game_id)  # HACK

        # Final rollout. Probably partial.
        for player in players.values():
            await player.rollout()

        # TODO(tzaman): the worldstate ends when game is over. the worldstate doesn't have info
        # about who won the game: so we need to get info from that somehow

        logger.info('Game finished.')


async def main(rmq_host, rmq_port, rollout_size, max_dota_time, latest_model_prob, initial_model):
    logger.info('main(rmq_host={}, rmq_port={})'.format(rmq_host, rmq_port))
    # RMQ
    rmq_connection = pika.BlockingConnection(pika.ConnectionParameters(host=rmq_host, port=rmq_port, heartbeat=300))
    experience_channel = rmq_connection.channel()
    experience_channel.queue_declare(queue=EXPERIENCE_QUEUE_NAME)

    weight_store.ready = asyncio.Event(loop=asyncio.get_event_loop())

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

    config = GameConfig(
        ticks_per_observation=TICKS_PER_OBSERVATION,
        host_timescale=HOST_TIMESCALE,
        host_mode=HOST_MODE,
        game_mode=DOTA_GAMEMODE_1V1MID,
    )

    game = Game(config=config, dota_service=dota_service, experience_channel=experience_channel,
                rollout_size=rollout_size, max_dota_time=max_dota_time,
                latest_model_prob=latest_model_prob)

    for i in range(0, N_GAMES):
        logger.info('=== Starting Game {}.'.format(i))
        game_id = str(datetime.now().strftime('%b%d_%H-%M-%S'))
        try:
            await game.play(game_id=game_id)
        except:
            traceback.print_exc()
            return

    channel_dota.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", type=str, help="mq ip", default='127.0.0.1')
    parser.add_argument("--port", type=int, help="mq port", default=5672)
    parser.add_argument("--rollout-size", type=int, help="size of each rollout (steps)", default=256)
    parser.add_argument("--max-dota-time", type=int, help="Maximum in-game (dota) time of a game before restarting", default=600)
    parser.add_argument("-l", "--log", dest="log_level", help="Set the logging level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument("--model", type=str, help="Initial model to immediatelly start")
    parser.add_argument("--use-latest-model-prob", type=float,
                        help="Probability of using the latest model. Otherwise some old one is chosen if available.", default=1.0)
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    loop = asyncio.get_event_loop()
    coro = main(rmq_host=args.ip, rmq_port=args.port, rollout_size=args.rollout_size,
                max_dota_time=args.max_dota_time, latest_model_prob=args.use_latest_model_prob,
                initial_model=args.model)
    loop.run_until_complete(coro)
