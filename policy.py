import logging
from pprint import pformat, pprint
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


TICKS_PER_OBSERVATION = 15 # HACK!
# N_DELAY_ENUMS = 5  # HACK!

Action = namedtuple('Action', ['sample', 'probs', 'log_prob'])
REWARD_KEYS = ['win', 'xp', 'hp', 'kills', 'death', 'lh', 'denies', 'dist']  # HACK! use from agent!

class Policy(nn.Module):

    TICKS_PER_SECOND = 30
    MAX_MOVE_SPEED = 550
    MAX_MOVE_IN_OBS = (MAX_MOVE_SPEED / TICKS_PER_SECOND) * TICKS_PER_OBSERVATION
    N_MOVE_ENUMS = 9
    MOVE_ENUMS = np.arange(N_MOVE_ENUMS, dtype=np.float32) - int(N_MOVE_ENUMS / 2)
    MOVE_ENUMS *= MAX_MOVE_IN_OBS / (N_MOVE_ENUMS - 1) * 2
    OBSERVATIONS_PER_SECOND = TICKS_PER_SECOND / TICKS_PER_OBSERVATION

    def __init__(self):
        super().__init__()

        self.affine_env = nn.Linear(3, 128)

        self.affine_unit_basic_stats = nn.Linear(8, 128)

        self.affine_unit_ah = nn.Linear(128, 128)
        self.affine_unit_eh = nn.Linear(128, 128)
        self.affine_unit_anh = nn.Linear(128, 128)
        self.affine_unit_enh = nn.Linear(128, 128)

        self.affine_pre_rnn = nn.Linear(640, 128)
        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

        # self.ln = nn.LayerNorm(128)

        # Heads
        self.affine_head_enum = nn.Linear(128, 3)
        self.affine_move_x = nn.Linear(128, self.N_MOVE_ENUMS)
        self.affine_move_y = nn.Linear(128, self.N_MOVE_ENUMS)
        # self.affine_head_delay = nn.Linear(128, N_DELAY_ENUMS)
        self.affine_unit_attention = nn.Linear(128, 128)

    def single(self, hidden, **kwargs):
        """Inputs a single element of a sequence."""
        for k in kwargs:
            kwargs[k] = kwargs[k].unsqueeze(0).unsqueeze(0)
        return self.__call__(**kwargs, hidden=hidden)

    def sequence(self, hidden, **kwargs):
        """Inputs a single sequence."""
        for k in kwargs:
            kwargs[k] = kwargs[k].unsqueeze(0)
        return self.__call__(**kwargs, hidden=hidden)

    INPUT_KEYS = ['env', 'allied_heroes', 'enemy_heroes', 'allied_nonheroes', 'enemy_nonheroes']

    def forward(self, env, allied_heroes, enemy_heroes, allied_nonheroes, enemy_nonheroes, hidden):
        """Input as batch."""
        logger.debug('policy(inputs=\n{}'.format(
            pformat({'env': env,
            'allied_heroes': allied_heroes,
            'enemy_heroes': enemy_heroes,
            'allied_nonheroes': allied_nonheroes,
            'enemy_nonheroes': enemy_nonheroes,
            })))

        # Environment.
        env = F.relu(self.affine_env(env))  # (b, s, n)

        # Allied Heroes.
        ah_basic = F.relu(self.affine_unit_basic_stats(allied_heroes))
        ah_embedding = self.affine_unit_ah(ah_basic)  # (b, s, units, n)
        ah_embedding_max, _ = torch.max(ah_embedding, dim=2)  # (b, s, n)

        # Enemy Heroes.
        eh_basic = F.relu(self.affine_unit_basic_stats(enemy_heroes))
        eh_embedding = self.affine_unit_eh(eh_basic)  # (b, s, units, n)
        eh_embedding_max, _ = torch.max(eh_embedding, dim=2)  # (b, s, n)

        # Allied Non-Heroes.
        anh_basic = F.relu(self.affine_unit_basic_stats(allied_nonheroes))
        anh_embedding = self.affine_unit_anh(anh_basic)  # (b, s, units, n)
        anh_embedding_max, _ = torch.max(anh_embedding, dim=2)  # (b, s, n)

        # Enemy Non-Heroes.
        enh_basic = F.relu(self.affine_unit_basic_stats(enemy_nonheroes))
        enh_embedding = self.affine_unit_enh(enh_basic)  # (b, s, units, n)
        enh_embedding_max, _ = torch.max(enh_embedding, dim=2)  # (b, s, n)

        # Create the full unit embedding
        unit_embedding = torch.cat((ah_embedding, eh_embedding, anh_embedding, enh_embedding), dim=2)  # (b, s, units, n)
        unit_embedding = torch.transpose(unit_embedding, dim0=3, dim1=2)  # (b, s, units, n) -> (b, s, n, units)

        # Combine for LSTM.
        x = torch.cat((env, ah_embedding_max, eh_embedding_max, anh_embedding_max, enh_embedding_max), dim=2)  # (b, s, n)

        x = F.relu(self.affine_pre_rnn(x))  # (b, s, n)

        # TODO(tzaman) Maybe add parameter noise here.
        # x = self.ln(x)

        # LSTM
        x, hidden = self.rnn(x, hidden)  # (b, s, n)
        
        # Heads.
        action_scores_x = self.affine_move_x(x)
        action_scores_y = self.affine_move_y(x)
        action_scores_enum = self.affine_head_enum(x)
        # action_delay_enum = self.affine_head_delay(x)
        action_unit_attention = self.affine_unit_attention(x)  # (b, s, n)

        action_unit_attention = action_unit_attention.unsqueeze(2)  # (b, s, n) ->  (b, s, 1, n)

        action_target_unit = torch.matmul(action_unit_attention, unit_embedding)   # (b, s, 1, n) * (b, s, n, units) = (b, s, 1, units)

        action_target_unit = action_target_unit.squeeze(2)  # (b, s, 1, units) -> (b, s, units)

        # Action space noise?
        # action_scores_enum += (torch.randn(action_scores_enum.shape) * 1)

        action_dict = dict(
            enum=F.softmax(action_scores_enum, dim=2),  # (b, s, 3)
            x=F.softmax(action_scores_x, dim=2),  # (b, s, 9)
            y=F.softmax(action_scores_y, dim=2),  # (b, s, 9)
            # delay=F.softmax(action_delay_enum, dim=2),
            target_unit=F.softmax(action_target_unit, dim=2),  # (b, s, units)
        )
        return action_dict, hidden

    @staticmethod
    def flatten_action_dict(inputs):
        """Flattens dicts with probabilities per actions to a dense (probability) tensor"""
        return torch.cat([inputs['enum'], inputs['x'], inputs['y'], inputs['target_unit']], dim=2)

    @staticmethod
    def flatten_selections(inputs):
        """Flatten a dict with a (n-multi)action selection(s) into a 'n-hot' 1D tensor"""
        d = {'enum': 3, 'x': 9, 'y': 9, 'target_unit': 38}  # 1+5+16+16=38
        t = torch.zeros(sum(d.values()), dtype=torch.uint8)
        i = 0
        for key, val in d.items():
            if key in inputs:
                t[i + inputs[key]] = 1
            i += val
        return t

    @staticmethod
    def sample_action(probs):
        return Categorical(probs).sample()

    @staticmethod
    def action_log_prob(probs, sample):
        return Categorical(probs).log_prob(sample)

    @classmethod
    def select_actions(cls, head_prob_dict):
        # From all heads, select actions.
        action_dict = {}
        # First select the high-level action.
        action_dict['enum'] = cls.sample_action(head_prob_dict['enum'])

        if action_dict['enum'] == 0:  # Nothing
            pass
        elif action_dict['enum'] == 1:  # Move
            action_dict['x'] = cls.sample_action(head_prob_dict['x'])
            action_dict['y'] = cls.sample_action(head_prob_dict['y'])
        elif action_dict['enum'] == 2:  # Attack
            if head_prob_dict['target_unit'].size(1) != 0:
                action_dict['target_unit'] = cls.sample_action(head_prob_dict['target_unit'])
        else:
            ValueError("Invalid Action Selection.")

        return action_dict

    @staticmethod
    def mask_heads(head_prob_dict, unit_handles):
        """Mask the head with possible actions."""
        # Mark your own unit as invalid
        invalid_units = unit_handles == -1
        invalid_units[0] = 1 # The 'self' hero can never be targetted.
        if invalid_units.all():
            # All units invalid, so we cannot choose the high-level attack head:
            head_prob_dict['enum'][0, 0, 2] = 0.
        head_prob_dict['target_unit'][0, 0, invalid_units] = 0.
        return head_prob_dict

    @classmethod
    def action_probs(cls, head_prob_dict, action_dict):
        # Given heads (probabilities) and actions distinctly selected from those, join these
        # pieces of information to yield: (1) the action (2) the prob (3) the logprob
        action_probs = {}
        for k, v in action_dict.items():
            action_probs[k] = Action(
                sample=v,
                probs=head_prob_dict[k],
                log_prob=cls.action_log_prob(probs=head_prob_dict[k], sample=v),
                )
        return action_probs


class RndModel(torch.nn.Module):

    def __init__(self, requires_grad):
        super().__init__()
        self.affine1 = torch.nn.Linear(10, 64)
        self.affine2 = torch.nn.Linear(64, 64)
        self.affine3 = torch.nn.Linear(64, 64)
        self.affine4 = torch.nn.Linear(64, 64)
        self.requires_grad = requires_grad

    def forward(self, env, allied_heroes, enemy_heroes, allied_nonheroes, enemy_nonheroes):
        if allied_heroes.size(0) == 0:  # HACK: Dead hero.
            allied_heroes = torch.zeros(1, 8)
        inputs = torch.cat([env.view(-1), allied_heroes.view(-1)])
        x = F.relu(self.affine1(inputs))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))
        return x