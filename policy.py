import logging
from pprint import pformat, pprint
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


logger = logging.getLogger(__name__)

TICKS_PER_OBSERVATION = 10 # HACK!
# N_DELAY_ENUMS = 5  # HACK!

Action = namedtuple('Action', ['sample', 'probs', 'log_prob'])

class Policy(nn.Module):

    TICKS_PER_SECOND = 30
    MAX_MOVE_SPEED = 550
    MAX_MOVE_IN_OBS = (MAX_MOVE_SPEED / TICKS_PER_SECOND) * TICKS_PER_OBSERVATION
    N_MOVE_ENUMS = 9
    MOVE_ENUMS = np.arange(N_MOVE_ENUMS, dtype=np.float32) - int(N_MOVE_ENUMS / 2)
    MOVE_ENUMS *= MAX_MOVE_IN_OBS / (N_MOVE_ENUMS - 1) * 2

    def __init__(self):
        super().__init__()
        self.affine_loc = nn.Linear(2, 128)
        self.affine_env = nn.Linear(2, 128)

        self.affine_unit_enh1 = nn.Linear(3, 32)
        self.affine_unit_enh2 = nn.Linear(32, 128)

        self.affine_unit_anh1 = nn.Linear(3, 32)
        self.affine_unit_anh2 = nn.Linear(32, 128)

        self.affine_unit_h1 = nn.Linear(3, 32)
        self.affine_unit_h2 = nn.Linear(32, 128)

        self.affine_pre_rnn = nn.Linear(128*5, 128)

        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=1)
        self.ln = nn.LayerNorm(128)

        self.affine_move_x = nn.Linear(128, self.N_MOVE_ENUMS)
        self.affine_move_y = nn.Linear(128, self.N_MOVE_ENUMS)
        self.affine_head_enum = nn.Linear(128, 3)
        # self.affine_head_delay = nn.Linear(128, N_DELAY_ENUMS)
        self.affine_unit_attention = nn.Linear(128, 128)


    def get_grad_dict(self):
        print('Policy::get_grad_dict()')
        grad_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                grad_dict[name] = param.grad
        return grad_dict

    def set_grad_dict(self, grad_dict):
        print('Policy::set_grad_dict()')
        for name, param in self.named_parameters():
            param.grad = grad_dict[name]

    def forward(self, loc, env, enemy_nonheroes, allied_nonheroes, enemy_heroes, hidden):
        logger.debug('policy(inputs=\n{}'.format(
            pformat({'loc': loc, 'env': env, 'enemy_nonheroes': enemy_nonheroes,
            'allied_nonheroes': allied_nonheroes})))

        loc = F.relu(self.affine_loc(loc))
        env = F.relu(self.affine_env(env))

        unit_embedding = torch.empty([0, 128])

        enh_embedding = []
        for unit_m in enemy_nonheroes:
            enh1 = F.relu(self.affine_unit_enh1(unit_m))
            enh2 = self.affine_unit_enh2(enh1)
            enh_embedding.append(enh2)

        if enh_embedding:
            # Create the variable length embedding for use in LSTM and attention head.
            enh_embedding = torch.stack(enh_embedding)  # shape: (n_units, 128)
            # We max over unit dim to have a fixed output shape bc the LSTM needs to learn about these units.
            enh_embedding_max, _ = torch.max(enh_embedding, dim=0)  # shape: (128,)
            enh_embedding_max = enh_embedding_max.unsqueeze(0) # shape: (1, 128)
            unit_embedding = torch.cat((unit_embedding, enh_embedding), 0)  # (n, 128)
        else:
            enh_embedding_max = torch.zeros(1, 128)


        anh_embedding = []
        for unit_m in allied_nonheroes:
            anh1 = F.relu(self.affine_unit_anh1(unit_m))
            anh2 = self.affine_unit_anh2(anh1)
            anh_embedding.append(anh2)

        if anh_embedding:
            # Create the variable length embedding for use in LSTM and attention head.
            anh_embedding = torch.stack(anh_embedding)  # shape: (n_units, 128)
            # We max over unit dim to have a fixed output shape bc the LSTM needs to learn about these units.
            anh_embedding_max, _ = torch.max(anh_embedding, dim=0)  # shape: (128,)
            anh_embedding_max = anh_embedding_max.unsqueeze(0)
            unit_embedding = torch.cat((unit_embedding, anh_embedding), 0)  # (n, 128)
        else:
            anh_embedding_max = torch.zeros(1, 128)


        h_embedding = []
        for unit_m in enemy_heroes:
            h1 = F.relu(self.affine_unit_h1(unit_m))
            h2 = self.affine_unit_h2(h1)
            h_embedding.append(h2)

        if h_embedding:
            # Create the variable length embedding for use in LSTM and attention head.
            h_embedding = torch.stack(h_embedding)  # shape: (n_units, 128)
            # We max over unit dim to have a fixed output shape bc the LSTM needs to learn about these units.
            h_embedding_max, _ = torch.max(h_embedding, dim=0)  # shape: (128,)
            h_embedding_max = h_embedding_max.unsqueeze(0)
            unit_embedding = torch.cat((unit_embedding, h_embedding), 0)  # (n, 128)
        else:
            h_embedding_max = torch.zeros(1, 128)


        # Combine for LSTM.
        x = torch.cat((loc, env, enh_embedding_max, anh_embedding_max, h_embedding_max), 1)  # (512,)

        x = F.relu(self.affine_pre_rnn(x))

        # TODO(tzaman) Maybe add parameter noise here.
        x = self.ln(x)

        # LSTM
        x, hidden = self.rnn(x.unsqueeze(1), hidden)
        x = x.squeeze(1)

        # Heads.
        action_scores_x = self.affine_move_x(x)
        action_scores_y = self.affine_move_y(x)
        action_scores_enum = self.affine_head_enum(x)
        # action_delay_enum = self.affine_head_delay(x)
        action_unit_attention = self.affine_unit_attention(x)  # shape: (1, 256)
        action_unit_attention = torch.mm(action_unit_attention, unit_embedding.t())  # shape (1, n)

        action_dict = dict(
            enum=F.softmax(action_scores_enum, dim=1),
            x=F.softmax(action_scores_x, dim=1),
            y=F.softmax(action_scores_y, dim=1),
            # delay=F.softmax(action_delay_enum, dim=1),
            target_unit=F.softmax(action_unit_attention, dim=1),
        )

        # TODO(tzaman): what is the correct way to handle invalid actions like below?
        if action_dict['target_unit'].shape[1] == 0:
            # If there are no units to target, we cannot perform 'action'
            # TODO(tzaman): come up with something nice and generic here.
            x = action_dict['enum'].clone()
            x[0][2] = 0
            action_dict['enum'] = x

        return action_dict, hidden


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
            action_dict['target_unit'] = cls.sample_action(head_prob_dict['target_unit'])
        else:
            ValueError("Invalid Action Selection.")

        return action_dict

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
            
