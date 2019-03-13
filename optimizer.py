from collections import Counter
from collections import deque
from datetime import datetime
import argparse
import copy
import gc
import io
import logging
import os
import pickle
import re
import socket
import time
import random

from google.cloud import storage
from tensorboardX import SummaryWriter
import numpy as np
import pika
from scipy.signal import lfilter
import torch
import torch.distributed as dist

from distributed import DistributedDataParallelSparseParamCPU
from dotaservice.protos.DotaService_pb2 import TEAM_DIRE, TEAM_RADIANT
from policy import Policy
from policy import REWARD_KEYS

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# torch.set_printoptions(profile="full")

torch.manual_seed(7)
random.seed(7)
np.random.seed(7)

eps = np.finfo(np.float32).eps.item()


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def is_master():
    if is_distributed():
        return torch.distributed.get_rank() == 0
    else:
        return True


def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype(np.float32)


def advantage_returns(rewards, values, gamma, lam):
    """Compute the advantage and returns from rewards and values."""
    # GAE-Lambda advantage calculation.
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    advantages = discount(deltas, gamma * lam)
    # Compute rewards-to-go (targets for the value function).
    returns = discount(rewards, gamma)[:-1]
    return advantages, returns


class MessageQueue:
    EXPERIENCE_QUEUE_NAME = 'experience'
    MODEL_EXCHANGE_NAME = 'model'
    MAX_RETRIES = 10

    def __init__(self, host, port, prefetch_count, use_model_exchange):
        """
        Args:
            prefetch_count (int): Amount of messages to prefetch. Settings this variable too
                high can result in blocked pipes that time out.
        """
        self._params = pika.ConnectionParameters(
            host=host,
            port=port,
            heartbeat=300,
        )
        self.prefetch_count = prefetch_count
        self.use_model_exchange = use_model_exchange

        self._conn = None
        self._xp_channel = None
        self._model_exchange = None

    def process_events(self):
        try:
            self._conn.process_data_events()
        except:
            pass

    def connect(self):
        if not self._conn or self._conn.is_closed:
            # RMQ.
            for i in range(10):
                try:
                    self._conn = pika.BlockingConnection(self._params)
                except pika.exceptions.ConnectionClosed:
                    logger.error('Connection to RMQ failed. retring. ({}/{})'.format(i, self.MAX_RETRIES))
                    time.sleep(5)
                    continue
                else:
                    logger.info('Connected to RMQ')
                    break

            # Experience channel.
            self._xp_channel = self._conn.channel()
            self._xp_channel.basic_qos(prefetch_count=self.prefetch_count)
            self._xp_channel.queue_declare(queue=self.EXPERIENCE_QUEUE_NAME)

            # Model Exchange.
            if self.use_model_exchange:
                self._model_exchange = self._conn.channel()
                self._model_exchange.exchange_declare(
                    exchange=self.MODEL_EXCHANGE_NAME,
                    exchange_type='x-recent-history',
                    arguments={'x-recent-history-length': 1},
                )

    @property
    def xp_queue_size(self):
        try:
            res = self._xp_channel.queue_declare(queue=self.EXPERIENCE_QUEUE_NAME, passive=True)
            return res.method.message_count
        except:
            return None

    def process_data_events(self):
        # Sends heartbeat, might keep conn healthier.
        try:
            self._conn.process_data_events()
        except:  # Gotta catch em' all!
            pass

    def _publish_model(self, msg, hdr):
        self._model_exchange.basic_publish(
            exchange=self.MODEL_EXCHANGE_NAME,
            routing_key='',
            body=msg,
            properties=pika.BasicProperties(headers=hdr),
        )

    def publish_model(self, *args, **kwargs):
        try:
            self._publish_model(*args, **kwargs)
        except (pika.exceptions.ConnectionClosed, pika.exceptions.ChannelClosed):
            logger.error('reconnecting to queue')
            self.connect()
            self._publish_model(*args, **kwargs)

    def _consume_xp(self):
        method, properties, body = next(self._xp_channel.consume(
            queue=self.EXPERIENCE_QUEUE_NAME,
            no_ack=False,
        ))
        self._xp_channel.basic_ack(delivery_tag=method.delivery_tag)
        return method, properties, body

    def consume_xp(self):
        try:
            return self._consume_xp()
        except (pika.exceptions.ConnectionClosed, pika.exceptions.ChannelClosed):
            logger.error('reconnecting to queue')
            self.connect()
            return self._consume_xp()

    def close(self):
        if self._conn and self._conn.is_open:
            logger.info('closing queue connection')
            self._conn.close()

class Sequence:
    def __init__(self, game_id, weight_version, team_id, observations, actions, masks, values, rewards, hidden, log_probs_sel):
        self.game_id = game_id
        self.weight_version = weight_version
        self.team_id = team_id
        self.observations = observations
        self.actions = actions
        self.masks = masks
        self.rewards = rewards
        self.values = values
        self.hidden = hidden
        self.log_probs_sel = log_probs_sel
        # Below are to be assigned later.
        self.advantages = None
        self.returns = None


def all_gather(t):
    _t = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(_t, t)
    return torch.cat(_t)


class DotaOptimizer:

    MODEL_FILENAME_FMT = "model_%09d.pt"
    BUCKET_NAME = 'dotaservice'
    RUNNING_NORM_FACTOR = 0.99  # Updates with every received rollout.
    MODEL_HISTOGRAM_FREQ = 128
    MAX_GRAD_NORM = 0.5
    SPEED_KEY = 'steps per s'

    def __init__(self, rmq_host, rmq_port, epochs, min_seq_per_epoch, seq_len,
                 learning_rate, checkpoint, pretrained_model, mq_prefetch_count, log_dir,
                 entropy_coef, vf_coef, run_local):
        super().__init__()
        self.rmq_host = rmq_host
        self.rmq_port = rmq_port
        self.epochs = epochs
        self.min_seq_per_epoch = min_seq_per_epoch
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.checkpoint = checkpoint
        self.mq_prefetch_count = mq_prefetch_count
        self.iteration_start = 1
        self.policy_base = Policy()
        self.log_dir = log_dir
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.run_local = run_local

        self.iterations = 10000
        self.e_clip = 0.1

        self.running_mean = {int(team_id): None for team_id in [TEAM_RADIANT, TEAM_DIRE]}
        self.running_std = {int(team_id): None for team_id in [TEAM_RADIANT, TEAM_DIRE]}

        if self.checkpoint:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            logger.info('Checkpointing to: {}'.format(self.log_dir))
            # if we are not running locally, set bucket
            if not self.run_local:
                client = storage.Client()
                self.bucket = client.get_bucket(self.BUCKET_NAME)

            # First, check if logdir exists.
            latest_model = self.get_latest_model(prefix=self.log_dir)
            # If there's a model in here, we resume from there
            if latest_model is not None:
                logger.info('Found a latest model in pretrained dir: {}'.format(latest_model))
                self.iteration_start = self.iteration_from_model_filename(filename=latest_model) + 1
                if pretrained_model is not None:
                    logger.warning('Overriding pretrained model by latest model.')
                pretrained_model = latest_model

            # if we are not running locally, pull down model
            if not self.run_local:
                if pretrained_model is not None:
                    logger.info('Downloading: {}'.format(pretrained_model))
                    model_blob = self.bucket.get_blob(pretrained_model)
                    # TODO(tzaman): Download to BytesIO and supply to torch in that way.
                    pretrained_model = '/tmp/model.pt'
                    model_blob.download_to_filename(pretrained_model)

        if pretrained_model is not None:
            self.policy_base.load_state_dict(torch.load(pretrained_model), strict=False)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.policy = DistributedDataParallelSparseParamCPU(self.policy_base)
        else:
            self.policy = self.policy_base

        self.policy_old = copy.deepcopy(self.policy)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.time_last_step = time.time()

        self.mq = MessageQueue(host=self.rmq_host, port=self.rmq_port,
                               prefetch_count=mq_prefetch_count,
                               use_model_exchange=self.checkpoint)
        self.mq.connect()

        # Upload initial model before any step is taken, and only if we're not resuming.
        if self.iteration_start == 1:
            self.upload_model(version=0)


    @staticmethod
    def iteration_from_model_filename(filename):
        x = re.search('(\d+)(?=.pt)', filename)
        return int(x.group(0))

    def get_latest_model(self, prefix):
        if not self.run_local:
            blobs = list(self.bucket.list_blobs(prefix=prefix))
        else:
            blobs = [f for f in os.listdir(prefix) if os.path.isfile(f)]

        if not blobs:
            # Directory does not exist, or no files in directory.
            return None
        else:
            fns = [x.name for x in blobs if x.name[-3:] == '.pt']
            if not fns:
                # No relevant files in directory.
                return None
            fns.sort()
            latest_model = fns[-1]
            return latest_model

    @property
    def events_filename(self):
        return self.writer.file_writer.event_writer._ev_writer._file_name

    def get_rollout(self):
        # TODO(tzaman): make a rollout object
        method, properties, body = self.mq.consume_xp()
        data = pickle.loads(body)
        reward_sum = np.sum(data['rewards'])
        rollout_len = data['rewards'].shape[0]
        version = data['weight_version']
        canvas = data['canvas']

        # Compute rewards per topic, reduce-sum down the sequences.
        subrewards = data['rewards'].sum(axis=0)

        return data, subrewards, rollout_len, version, canvas

    def update_running_mean_std(self, x, team_id):
        mean_reward = x.mean()
        mean_std = x.std()
        if self.running_mean[team_id] is None:  #  First ever step
            self.running_mean[team_id] = mean_reward
            self.running_std[team_id] = mean_std
        else:
            self.running_mean[team_id] = self.running_mean[team_id] * self.RUNNING_NORM_FACTOR + mean_reward * (1 - self.RUNNING_NORM_FACTOR)
            self.running_std[team_id] = self.running_std[team_id] * self.RUNNING_NORM_FACTOR + mean_std * (1 - self.RUNNING_NORM_FACTOR)

    def experiences_from_rollout(self, data):
        # TODO(tzaman): The rollout can consist out of multiple viable sequences.
        # These should be padded and then sliced into separate experiences.
        observations = data['observations']
        masks = data['masks']
        actions = data['actions']
        rewards = data['rewards']

        rollout_len = data['rewards'].shape[0]
        logger.debug('rollout_len={}'.format(rollout_len))

        start = time.time()

        sequences = []
        hidden = self.policy.init_hidden()
        values = []
        rewards_sum = []
        slice_indices = range(0, rollout_len, self.seq_len)
        for i1 in slice_indices:
            pad = 0
            # Check if this slice requires padding.
            if rollout_len - i1 < self.seq_len:
                pad = self.seq_len - (rollout_len - i1)
            i2 = i1 + self.seq_len - pad
            logger.debug('Slice[{}:{}], pad={}'.format(i1, i2, pad))

            # Slice out the relevant parts.
            s_observations = {}
            for key, val in observations.items():
                s_observations[key] = val[i1:i2, :]

            s_masks = {}
            for key, val in masks.items():
                s_masks[key] = val[i1:i2, :]

            s_actions = {}
            for key, val in actions.items():
                s_actions[key] = val[i1:i2, :]

            s_rewards = rewards[i1:i2]

            if pad:
                dim_pad = {
                    1: (0, pad),
                    2: (0, 0, 0, pad),
                    3: (0, 0, 0, 0, 0, pad),
                }
                for key, val in s_observations.items():
                    s_observations[key] = torch.nn.functional.pad(val, dim_pad[val.dim()], mode='constant', value=0).detach()

                for key, val in s_masks.items():
                    s_masks[key] = torch.nn.functional.pad(val, pad=dim_pad[val.dim()], mode='constant', value=0).detach()

                for key, val in s_actions.items():
                    s_actions[key] = torch.nn.functional.pad(val, pad=dim_pad[val.dim()], mode='constant', value=0).detach()

                s_rewards = np.pad(s_rewards, ((0, pad), (0, 0)), mode='constant')

            input_hidden = hidden
            head_logits_dict, s_values, hidden = self.policy.sequence(**s_observations, hidden=input_hidden)

            log_probs_sel = {}
            for key in head_logits_dict:
                # TODO(tzaman): USE FOR PPO
                log_probs = Policy.masked_softmax(logits=head_logits_dict[key], mask=s_masks[key].unsqueeze(0))
                log_probs_sel[key] = torch.masked_select(input=log_probs, mask=s_actions[key]).detach()

            # The values and rewards are gathered here over all sequences, because the values are
            # cumulative, and therefore need the first step from each next sequence. To optimize this,
            # we gather them here, and process these after the loop, and add them to the Sequence
            # object later.
            values.append(s_values)
            rewards_sum.append(np.sum(s_rewards, axis=1).ravel())

            sequence = Sequence(
                game_id=data['game_id'],
                weight_version=data['weight_version'],
                team_id=data['team_id'],
                observations=s_observations,
                actions=s_actions,
                masks=s_masks,
                values=s_values.detach(),
                rewards=s_rewards,
                hidden=input_hidden.detach(),
                log_probs_sel=log_probs_sel,
            )
            sequences.append(sequence)

        # TODO(tzaman): For now, we assume we are always presented with full and terminated sequences.
        # This is why we append a zero here, so the advantage and return computation works efficiently.
        # In the future, if we support receiving non-terminated sequences, we need one more state and
        # reward than we have action.
        values = torch.cat(values).numpy().ravel()
        values = np.append(values, np.array(0., dtype=np.float32))
        rewards_sum = np.concatenate(rewards_sum)
        rewards_sum = np.append(rewards_sum, np.array(0., dtype=np.float32))
        advantages, returns = advantage_returns(rewards=rewards_sum, values=values, gamma=0.98, lam=0.97)

        # Split the advantages and returns back up into their respective sequences.
        advantages = np.split(advantages, len(sequences))
        returns = np.split(returns, len(sequences))
        for s, a, r in zip(sequences, advantages, returns):
            s.advantages = torch.from_numpy(a)
            s.returns = torch.from_numpy(r)

        return sequences

    @staticmethod
    def list_of_dicts_to_dict_of_lists(x):
        return {k: torch.stack([d[k] for d in x]) for k in x[0]}

    def run(self):
        for it in range(self.iteration_start, self.iterations):
            logger.info('iteration {}/{}'.format(it, self.iterations))

            # First grab a bunch of experiences
            experiences = []
            subrewards = []
            rollout_lens = []
            weight_ages = []
            canvas = None  # Just save only the last canvas.
            while len(experiences) < self.min_seq_per_epoch:
                logger.debug(' adding experience @{}/{}'.format(len(experiences), self.min_seq_per_epoch))

                # Get new experiences from a new rollout.
                with torch.no_grad():
                    rollout, rollout_subrewards, rollout_len, weight_version, canvas = self.get_rollout()
                    rollout_experiences = self.experiences_from_rollout(data=rollout)

                experiences.extend(rollout_experiences)

                subrewards.append(rollout_subrewards)
                rollout_lens.append(rollout_len)
                weight_ages.append(it - weight_version)

            losses = []
            entropies = []
            grad_norms = []
            for ep in range(self.epochs):
                logger.info(' epoch {}/{}'.format(ep + 1, self.epochs))
                self.mq.process_data_events()
                loss_d, entropy_d, grad_norm_d = self.train(experiences=experiences)
                losses.append(loss_d)
                entropies.append(entropy_d)
                grad_norms.append(grad_norm_d)

            # Set the new policy as the old one.
            self.policy_old.load_state_dict(self.policy.state_dict())

            losses = self.list_of_dicts_to_dict_of_lists(losses)
            loss = losses['loss'].mean()

            entropies = self.list_of_dicts_to_dict_of_lists(entropies)
            entropy = torch.stack(list(entropies.values())).sum(dim=0).mean()

            grad_norms = self.list_of_dicts_to_dict_of_lists(grad_norms)

            n_steps = len(experiences) * self.seq_len
            steps_per_s = n_steps / (time.time() - self.time_last_step)
            self.time_last_step = time.time()

            subrewards_per_sec = np.stack(subrewards) / n_steps * Policy.OBSERVATIONS_PER_SECOND
            rollout_rewards = subrewards_per_sec.sum(axis=1)
            reward_dict = dict(zip(REWARD_KEYS, subrewards_per_sec.sum(axis=0)))
            reward_per_sec = rollout_rewards.sum()

            rollout_lens = torch.tensor(rollout_lens, dtype=torch.float32)
            avg_rollout_len = rollout_lens.mean()

            weight_ages = torch.tensor(weight_ages, dtype=torch.float32)
            avg_weight_age = weight_ages.mean()

            metrics = {
                self.SPEED_KEY: steps_per_s,
                'reward_per_sec/sum': reward_per_sec,
                'loss/sum': loss,
                'loss/policy': losses['policy_loss'].mean(),
                'loss/entropy': losses['entropy_loss'].mean(),
                'loss/value': losses['value_loss'].mean(),
                'entropy': entropy,
                'avg_rollout_len': avg_rollout_len,
                'avg_weight_age': avg_weight_age,
            }

            for team_id in self.running_mean:
                if self.running_mean[team_id] is not None:
                    metrics['rewards/running_mean_{}'.format(team_id)] = self.running_mean[team_id]

            for team_id in self.running_std:
                if self.running_std[team_id] is not None:
                    metrics['rewards/running_std_{}'.format(team_id)] = self.running_std[team_id]

            for k, v in entropies.items():
                metrics['entropy/{}'.format(k)] = v.mean()

            for k, v in grad_norms.items():
                metrics['grad_norm/{}'.format(k)] = v.mean()

            for k, v in reward_dict.items():
                metrics['reward_per_sec/{}'.format(k)] = v

            logger.info('steps_per_s={:.2f}, avg_weight_age={:.1f}, reward_per_sec={:.4f}, loss={:.4f}, entropy={:.3f}'.format(
                steps_per_s, float(avg_weight_age), reward_per_sec, float(loss), float(entropy)))

            if self.checkpoint:
                # TODO(tzaman): re-introduce distributed metrics. See commits from december 2017.

                # Write metrics to events file.
                for name, metric in metrics.items():
                    self.writer.add_scalar(name, metric, it)
                
                # Add per-iteration histograms
                self.writer.add_histogram('losses', losses['loss'], it)
                # self.writer.add_histogram('entropies', entropies, it)
                self.writer.add_histogram('rollout_lens', rollout_lens, it)
                self.writer.add_histogram('weight_age', weight_ages, it)

                # Rewards histogram
                self.writer.add_histogram('rewards_per_sec_per_rollout', rollout_rewards, it)

                # Model
                if it % self.MODEL_HISTOGRAM_FREQ == 1:
                    for name, param in self.policy_base.named_parameters():
                        self.writer.add_histogram('param/' + name, param.clone().cpu().data.numpy(), it)
                        self.writer.add_image('canvas', canvas, it, dataformats='HWC')

                # RMQ Queue size.
                queue_size = self.mq.xp_queue_size
                if queue_size is not None:
                    self.writer.add_scalar('mq_size', queue_size, it)

                # Upload events to GCS
                self.writer.file_writer.flush()  # Flush before uploading
                if not self.run_local:
                    blob = self.bucket.blob(self.events_filename)
                    blob.upload_from_filename(filename=self.events_filename)

                self.upload_model(version=it)

    def train(self, experiences):
        # Train on one epoch of data.
        # Experiences is a list of (padded) experience chunks.
        logger.debug('train(experiences=#{})'.format(len(experiences)))

        # Stack together all experiences.
        advantage = torch.stack([e.advantages for e in experiences])
        advantage = (advantage - advantage.mean()) / (advantage.std() + eps)
        advantage = advantage.detach()
        returns = torch.stack([e.returns for e in experiences]).detach()
        hidden = torch.cat([e.hidden for e in experiences], dim=1).detach()

        # The action mask contains the mask of the selected actions
        actions = {key: [] for key in Policy.OUTPUT_KEYS}
        for e in experiences:
            for key, val in e.actions.items():
                actions[key].append(val)
        for key in actions:
            actions[key] = torch.stack(actions[key])

        # The head mask contains the mask of the relevant heads, where a selection has taken place,
        # and includes only valid possible selections from those heads.
        masks = {key: [] for key in Policy.OUTPUT_KEYS}
        for e in experiences:
            for key, val in e.masks.items():
                masks[key].append(val)
        for key, val in masks.items():
            masks[key] = torch.stack(val)

        observations = {key: [] for key in Policy.INPUT_KEYS}
        for e in experiences:
            for key, val in e.observations.items():
                observations[key].append(val)
        for key, val in observations.items():
            observations[key] = torch.stack(val)

        # Notice there is no notion of loss masking here, this is unnessecary as we only work
        # use selected probabilties. E.g. when things were padded, nothing was selected, so no data.
        head_logits_dict, values, _ = self.policy(**observations, hidden=hidden)

        # Perform a masked softmax
        policy_loss = {}
        entropies = {}
        for key in head_logits_dict:
            log_probs = Policy.masked_softmax(logits=head_logits_dict[key], mask=masks[key])
            head_policy_loss = -log_probs * advantage.unsqueeze(-1)
            head_policy_loss_sel = torch.masked_select(input=head_policy_loss, mask=actions[key])
            policy_loss[key] = head_policy_loss_sel

            n_selections = head_policy_loss_sel.size(0)

            log_probs_sel = torch.masked_select(input=log_probs, mask=masks[key])
            probs_sel = torch.exp(log_probs_sel)
            if n_selections == 0:
                entropies[key] = torch.zeros([])
            else:
                entropies[key] = -(probs_sel * log_probs_sel).sum() / n_selections

        # Grab all the policy losses
        policy_loss = torch.cat([v for v in policy_loss.values()])
        policy_loss = policy_loss.mean()

        if self.entropy_coef > 0:
            entropy = torch.stack(list(entropies.values())).sum()
            entropy_loss = -self.entropy_coef * entropy
        else:
            entropy_loss = torch.tensor(0.)

        if self.vf_coef > 0:
            # Notice we don't have to remove zero-padded entries, as they give 0 loss.
            value_loss = 0.5 * (returns - values.squeeze(-1)).pow(2).mean()
            value_loss = self.vf_coef * value_loss
        else:
            value_loss = torch.tensor(0.)

        loss = policy_loss + entropy_loss + value_loss

        if torch.isnan(loss):
            raise ValueError('loss={}, policy_loss={}, entropy_loss={}, value_loss={}'.format(
                loss, policy_loss, entropy_loss, value_loss))

        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = self.mean_gradient_norm()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.MAX_GRAD_NORM)
        grad_norm_clipped = self.mean_gradient_norm()

        if torch.isnan(grad_norm):
            raise ValueError('grad_norm={}'.format(grad_norm))

        self.optimizer.step()
        losses = {
            'loss': loss,
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
            'value_loss': value_loss,
        }

        return losses, entropies, {'unclipped': grad_norm, 'clipped': grad_norm_clipped}

    def mean_gradient_norm(self):
        gs = []
        for p in list(filter(lambda p: p.grad is not None, self.policy.parameters())):
            gs.append(p.grad.data.norm(2))
        return torch.stack(gs).mean()

    def upload_model(self, version):
        if not is_master():
            # Only rank 0 uploads the model.
            return

        filename = self.MODEL_FILENAME_FMT % version
        rel_path = os.path.join(self.log_dir, filename)

        # Serialize the model.
        buffer = io.BytesIO()
        state_dict = self.policy_base.state_dict()
        torch.save(obj=state_dict, f=buffer)
        state_dict_b = buffer.getvalue()

        # Write model to file.
        with open(rel_path, 'wb') as f:
            f.write(state_dict_b)

        # Send to exchange.
        self.mq.publish_model(msg=state_dict_b, hdr={'version': version})

        # Upload to GCP.
        if not self.run_local:
            blob = self.bucket.blob(rel_path)
            blob.upload_from_string(data=state_dict_b)  # Model


def init_distribution(backend='gloo'):
    logger.info('init_distribution')
    assert 'WORLD_SIZE' in os.environ
    world_size = int(os.environ['WORLD_SIZE'])
    if world_size < 2:
        logger.warning('skipping distribution: world size too small ({})'.format(world_size))
        return
    torch.distributed.init_process_group(backend=backend)
    logger.info("Distribution initialized.")


def main(rmq_host, rmq_port, epochs, min_seq_per_epoch, seq_len, learning_rate,
         pretrained_model, mq_prefetch_count, log_dir, entropy_coef, vf_coef, run_local):
    logger.info('main(rmq_host={}, rmq_port={}, epochs={}, min_seq_per_epoch={},'
                ' seq_len={}, learning_rate={}, pretrained_model={}, mq_prefetch_count={}, entropy_coef={}, vf_coef={})'.format(
        rmq_host, rmq_port, epochs, min_seq_per_epoch, seq_len, learning_rate, pretrained_model, mq_prefetch_count,
        entropy_coef, vf_coef))

    # If applicable, initialize distributed training.
    if torch.distributed.is_available():
        init_distribution()
    else:
        logger.info('distribution unavailable')

    # Only the master should checkpoint.
    checkpoint = is_master()

    dota_optimizer = DotaOptimizer(
        rmq_host=rmq_host,
        rmq_port=rmq_port,
        epochs=epochs,
        min_seq_per_epoch=min_seq_per_epoch,
        seq_len=seq_len,
        learning_rate=learning_rate,
        checkpoint=checkpoint,
        pretrained_model=pretrained_model,
        mq_prefetch_count=mq_prefetch_count,
        log_dir=log_dir,
        entropy_coef=entropy_coef,
        vf_coef=vf_coef,
        run_local=run_local,
    )

    dota_optimizer.run()


def default_log_dir():
    return '{}_{}'.format(datetime.now().strftime('%b%d_%H-%M-%S'), socket.gethostname())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log-dir", type=str, help="log and job dir name", default=default_log_dir())
    parser.add_argument("--ip", type=str, help="mq ip", default='127.0.0.1')
    parser.add_argument("--port", type=int, help="mq port", default=5672)
    parser.add_argument("--epochs", type=int, help="amount of epochs", default=4)
    parser.add_argument("--min-seq-per-epoch", type=int, help="minimum amount of sequences per epoch."
        "This can be slightly more because we want to process full rollouts.", default=1024)
    parser.add_argument("--seq-len", type=int, help="sequence length (as one sample in a minibatch)."
        "This is also the length that will be (truncated) backpropped into.", default=16)
    parser.add_argument("--learning-rate", type=float, help="learning rate", default=5e-5)
    parser.add_argument("--entropy-coef", type=float, help="entropy coef (as proportional addition to the loss)", default=5e-4)
    parser.add_argument("--vf-coef", type=float, help="value fn coef (as proportional addition to the loss)", default=0.5)
    parser.add_argument("--pretrained-model", type=str, help="pretrained model file within gcs bucket", default=None)
    parser.add_argument("--mq-prefetch-count", type=int,
                        help="amount of experience messages to prefetch from mq", default=8)
    parser.add_argument("-l", "--log", dest="log_level", help="Set the logging level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument("--run-local", type=bool, help="set to true to run locally (not using GCP)", default=False)

    args = parser.parse_args()

    logger.setLevel(args.log_level)

    try:
        main(
            rmq_host=args.ip,
            rmq_port=args.port,
            epochs=args.epochs,
            min_seq_per_epoch=args.min_seq_per_epoch,
            seq_len=args.seq_len,
            learning_rate=args.learning_rate,
            pretrained_model=args.pretrained_model,
            mq_prefetch_count=args.mq_prefetch_count,
            log_dir=args.log_dir,
            entropy_coef=args.entropy_coef,
            vf_coef=args.vf_coef,
            run_local=args.run_local,
        )
    except KeyboardInterrupt:
        pass
