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

eps = np.finfo(np.float32).eps.item()

TICKS_PER_OBSERVATION = 15 # HACK!
# N_DELAY_ENUMS = 5  # HACK!

REWARD_KEYS = ['enemy', 'win', 'xp', 'hp', 'kills', 'death', 'lh', 'denies', 'tower_hp']


class MaskedCategorical():

    def __init__(self, log_probs, mask):
        self.log_probs = log_probs
        self.mask = mask
        self.masked_probs = torch.exp(log_probs).clone()
        self.masked_probs[~mask] = 0.
        # print('self.masked_probs=', self.masked_probs)

    def sample(self):
        return torch.multinomial(self.masked_probs[-1], num_samples=1)


class Policy(nn.Module):

    TICKS_PER_SECOND = 30
    MAX_MOVE_SPEED = 550
    MAX_MOVE_IN_OBS = (MAX_MOVE_SPEED / TICKS_PER_SECOND) * TICKS_PER_OBSERVATION
    N_MOVE_ENUMS = 9
    MOVE_ENUMS = np.arange(N_MOVE_ENUMS, dtype=np.float32) - int(N_MOVE_ENUMS / 2)
    MOVE_ENUMS *= MAX_MOVE_IN_OBS / (N_MOVE_ENUMS - 1) * 2
    OBSERVATIONS_PER_SECOND = TICKS_PER_SECOND / TICKS_PER_OBSERVATION
    MAX_UNITS = 1+5+16+16+1+1
    OUTPUT_KEYS = ['enum', 'x', 'y', 'target_unit']
    ACTION_OUTPUT_COUNTS = {'enum': 3, 'x': 9, 'y': 9, 'target_unit': MAX_UNITS}
    INPUT_KEYS = ['env', 'allied_heroes', 'enemy_heroes', 'allied_nonheroes', 'enemy_nonheroes',
                  'allied_towers', 'enemy_towers']


    def __init__(self):
        super().__init__()

        self.affine_env = nn.Linear(3, 128)

        self.affine_unit_basic_stats = nn.Linear(10, 128)

        self.affine_unit_ah = nn.Linear(128, 128)
        self.affine_unit_eh = nn.Linear(128, 128)
        self.affine_unit_anh = nn.Linear(128, 128)
        self.affine_unit_enh = nn.Linear(128, 128)
        self.affine_unit_ath = nn.Linear(128, 128)
        self.affine_unit_eth = nn.Linear(128, 128)

        self.affine_pre_rnn = nn.Linear(896, 256)
        self.rnn = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)

        # self.ln = nn.LayerNorm(128)

        # Heads
        self.affine_head_enum = nn.Linear(256, 3)
        self.affine_move_x = nn.Linear(256, self.N_MOVE_ENUMS)
        self.affine_move_y = nn.Linear(256, self.N_MOVE_ENUMS)
        # self.affine_head_delay = nn.Linear(128, N_DELAY_ENUMS)
        self.affine_unit_attention = nn.Linear(256, 128)
        self.affine_value = nn.Linear(256, 1)

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

    def forward(self, env, allied_heroes, enemy_heroes, allied_nonheroes, enemy_nonheroes,
                allied_towers, enemy_towers, hidden):
        """Input as batch."""

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

        # Allied Towers.
        ath_basic = F.relu(self.affine_unit_basic_stats(allied_towers))
        ath_embedding = self.affine_unit_ath(ath_basic)  # (b, s, units, n)
        ath_embedding_max, _ = torch.max(ath_embedding, dim=2)  # (b, s, n)

        # Enemy Towers.
        eth_basic = F.relu(self.affine_unit_basic_stats(enemy_towers))
        eth_embedding = self.affine_unit_eth(eth_basic)  # (b, s, units, n)
        eth_embedding_max, _ = torch.max(enh_embedding, dim=2)  # (b, s, n)

        # Create the full unit embedding
        unit_embedding = torch.cat((ah_embedding, eh_embedding, anh_embedding, enh_embedding, ath_embedding,
                                    eth_embedding), dim=2)  # (b, s, units, n)
        unit_embedding = torch.transpose(unit_embedding, dim0=3, dim1=2)  # (b, s, units, n) -> (b, s, n, units)

        # Combine for LSTM.
        x = torch.cat((env, ah_embedding_max, eh_embedding_max, anh_embedding_max, enh_embedding_max,
                       ath_embedding_max, eth_embedding_max), dim=2)  # (b, s, n)

        x = F.relu(self.affine_pre_rnn(x))  # (b, s, n)

        # TODO(tzaman) Maybe add parameter noise here.
        # x = self.ln(x)

        # LSTM
        x, hidden = self.rnn(x, hidden)  # (b, s, n)

        # Unit attention.
        unit_attention = self.affine_unit_attention(x)  # (b, s, n)
        unit_attention = unit_attention.unsqueeze(2)  # (b, s, n) ->  (b, s, 1, n)

        # Heads.
        action_scores_x = self.affine_move_x(x)
        action_scores_y = self.affine_move_y(x)
        action_scores_enum = self.affine_head_enum(x)
        # action_delay_enum = self.affine_head_delay(x)
        action_target_unit = torch.matmul(unit_attention, unit_embedding)   # (b, s, 1, n) * (b, s, n, units) = (b, s, 1, units)
        action_target_unit = action_target_unit.squeeze(2)  # (b, s, 1, units) -> (b, s, units)
        value = self.affine_value(x)  # (b, s, 1)

        d = {
            'enum': action_scores_enum,  # (b, s, 3)
            'x': action_scores_x,  # (b, s, 9)
            'y': action_scores_y,  # (b, s, 9)
            # delay=F.softmax(action_delay_enum, dim=2),
            'target_unit': action_target_unit # (b, s, units)
        }

        # Return
        return d, value, hidden

    @classmethod
    def masked_softmax(cls, logits, mask, dim=2):
        """Returns log-probabilities."""
        exp = torch.exp(logits)
        masked_exp = exp.clone()
        masked_exp[~mask] = 0.
        masked_sumexp = masked_exp.sum(dim, keepdim=True)
        logsumexp = torch.log(masked_sumexp)
        log_probs = logits - logsumexp
        return log_probs

    @classmethod
    def flatten_selections(cls, inputs):
        d = {}
        for key, count in cls.ACTION_OUTPUT_COUNTS.items():
            t = torch.zeros(count, dtype=torch.uint8)
            if key in inputs:
                t[inputs[key]] = 1
            d[key] = t
        return d

    @staticmethod
    def flatten_head(inputs, dim=2):
        return torch.cat(list(inputs.values()), dim=dim)

    @staticmethod
    def unpack_heads(inputs):
        return {
            'enum': inputs[..., :3],
            'x': inputs[..., 3:3+9],
            'y': inputs[..., 3+9:3+9+9],
            'target_unit': inputs[..., 3+9+9:],
        }

    @classmethod
    def flat_actions_to_headmask(cls, inputs):
        """Takes in flattened selections, then masks out full heads."""
        # TODO(tzaman): properly store all these magic numbers
        enum = inputs[:, :3]
        enumh = enum.any(dim=1, keepdim=True)
        enumh = enumh.repeat(1, 3)

        x = inputs[:, 3:3+9]
        xh = x.any(dim=1, keepdim=True)
        xh = xh.repeat(1, 9)

        y = inputs[:, 3+9:3+9+9]
        yh = y.any(dim=1, keepdim=True)
        yh = yh.repeat(1, 9)

        target_unit = inputs[:, 3+9+9:]
        target_unith = target_unit.any(dim=1, keepdim=True)
        target_unith = target_unith.repeat(1, cls.MAX_UNITS)

        # Put it back together
        return torch.cat([enumh, xh, yh, target_unith], dim=1)

    @classmethod
    def sample_action(cls, logits, mask)#, espilon=0.15):
        # TODO(tzaman): Have the sampler kind be user-configurable.
        # NOTE(tzaman): Epsilon-greedy is terrible e.g. in cases where the prob is every evenly
        # divided over choices: it will totally offset to one choice (e.g. esp with movement x or y).

        # # Below is episilon-random-uniform-stocastic
        # if torch.rand(1) < espilon:
        #     # `torch.multinomial` samples based on weights, so using the mask is already evenly
        #     # distributing the probabilities of all viable actions.
        #     sample = torch.multinomial(mask[-1].float(), num_samples=1)
        #     return sample
        # else:
        log_probs = cls.masked_softmax(logits=logits, mask=mask)
        sample = MaskedCategorical(log_probs=log_probs, mask=mask).sample()
        return sample

    @classmethod
    def select_actions(cls, heads_logits, masks):
        """From all heads, select actions."""
        action_dict = {}
        # First select the high-level action.
        action_dict['enum'] = cls.sample_action(heads_logits['enum'], mask=masks['enum'])

        if action_dict['enum'] == 0:  # Nothing
            pass
        elif action_dict['enum'] == 1:  # Move
            action_dict['x'] = cls.sample_action(heads_logits['x'], mask=masks['x'])
            action_dict['y'] = cls.sample_action(heads_logits['y'], mask=masks['y'])
        elif action_dict['enum'] == 2:  # Attack
            action_dict['target_unit'] = cls.sample_action(heads_logits['target_unit'], mask=masks['target_unit'])
        else:
            ValueError("Invalid Action Selection.")

        return action_dict

    @classmethod
    def head_masks(cls, selections):
        masks = {}
        for key, val in cls.ACTION_OUTPUT_COUNTS.items():
            fn = torch.ones if key in selections else torch.zeros
            masks[key] = fn(1, 1, val).byte()
        return masks

    @classmethod
    def action_masks(cls, unit_handles):
        """Mask the head with possible actions."""
        # Mark your own unit as invalid
        masks = {key: torch.ones(1, 1, val).byte() for key, val in cls.ACTION_OUTPUT_COUNTS.items()}
        valid_units = unit_handles != -1
        valid_units[0] = 0 # The 'self' hero can never be targetted.
        if not valid_units.any():
            # All units invalid, so we cannot choose the high-level attack head:
            masks['enum'][0, 0, 2] = 0
        masks['target_unit'][0, 0] = valid_units
        return masks

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


class RndModel(torch.nn.Module):

    def __init__(self, requires_grad):
        super().__init__()
        self.affine1 = torch.nn.Linear(10, 64)
        self.affine2 = torch.nn.Linear(64, 64)
        self.affine3 = torch.nn.Linear(64, 64)
        self.affine4 = torch.nn.Linear(64, 64)
        self.requires_grad = requires_grad

    def forward(self, env, allied_heroes, enemy_heroes, allied_nonheroes, enemy_nonheroes,
                allied_towers, enemy_towers):
        if allied_heroes.size(0) == 0:  # HACK: Dead hero.
            allied_heroes = torch.zeros(1, 7)
        inputs = torch.cat([env.view(-1), allied_heroes.view(-1)])
        x = F.relu(self.affine1(inputs))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))
        return x
