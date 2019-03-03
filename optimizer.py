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
import scipy.signal
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
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype(np.float32)


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
    def __init__(self, game_id, states, actions, masks, rewards, discounted_rewards, weight_version, team_id):
        self.game_id = game_id
        self.states = states
        self.actions = actions
        self.masks = masks
        self.rewards = rewards
        self.weight_version = weight_version
        self.team_id = team_id

        # Create a vector with the n-hot mask of actions.
        self.vec_action_mask = self.actions.view(-1).detach()
        self.vec_head_mask = self.masks.detach()

        # Count the amount of (multi-head) actions taken for each step.
        action_sum_per_step = torch.sum(self.actions, dim=1).view(-1).data.numpy()
        self.discounted_rewards = np.ravel(discounted_rewards)  # flat view
        # Repeat the rewards where a step has multiple actions, the reward gets repeated.
        self.vec_mh_rewards = torch.from_numpy(np.repeat(self.discounted_rewards, action_sum_per_step))

    def calculate_normalized_rewards(self, mean, std):
        self.vec_mh_rewards_norm = (self.vec_mh_rewards - mean) / (std + eps)

    # def calculate_old_probs(self, policy):
    #     head_logits_dict, _, _ = policy.sequence(**self.states, hidden=None)
    #     mask_dict = Policy.unpack_heads(self.masks)
    #     # Perform a masked softmax
    #     head_prob_dict = {}
    #     for key in head_logits_dict:
    #         head_prob_dict[key] = Policy.masked_softmax(x=head_logits_dict[key], mask=mask_dict[key])

    #     vec_probs_all = policy.flatten_head(head_prob_dict).view(-1)

    #     # Now mask the probs by the selection
    #     self.vec_old_probs = torch.masked_select(input=vec_probs_all, mask=self.vec_action_mask)
    #     self.vec_old_probs = self.vec_old_probs.detach()


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

    def __init__(self, rmq_host, rmq_port, epochs, seq_per_epoch, batch_size, seq_len,
                 learning_rate, checkpoint, pretrained_model, mq_prefetch_count, log_dir,
                 entropy_coef, vf_coef, run_local):
        super().__init__()
        self.rmq_host = rmq_host
        self.rmq_port = rmq_port
        self.epochs = epochs
        self.seq_per_epoch = seq_per_epoch
        self.batch_size = batch_size
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
        rollout_len = data['actions'].size(0)
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

        actions = data['actions']
        masks = data['masks']  # selected heads mask
        rewards = data['rewards']
        states = data['states']
        team_id = data['team_id']

        # If applicable, pad the rollout so we can cut into distinct sequences.
        rollout_len = data['actions'].size(0)
        pad = rollout_len % self.seq_len
        pad = 0 if pad == 0 else self.seq_len - pad
        n_sequences = rollout_len // self.seq_len if pad == 0 else rollout_len // self.seq_len + 1

        logger.debug('rollout_len={}, pad={}, n_sequences={}'.format(rollout_len, pad, n_sequences))

        if pad != 0:
            dim_pad = {
                1: (0, pad),
                2: (0, 0, 0, pad),
                3: (0, 0, 0, 0, 0, pad),
            }
            actions = torch.nn.functional.pad(actions, pad=dim_pad[actions.dim()], mode='constant', value=0)
            masks = torch.nn.functional.pad(masks, pad=dim_pad[masks.dim()], mode='constant', value=0)
            rewards = np.pad(rewards, ((0, pad), (0, 0)), mode='constant')

            for key in states:
                states[key] = torch.nn.functional.pad(states[key], dim_pad[states[key].dim()], mode='constant', value=0)

        # The advantage needs to be calculated here, in order to get information from the full
        # rollout.
        discounted_rewards = discount(x=np.sum(rewards, axis=1), gamma=0.98)

        self.update_running_mean_std(x=discounted_rewards, team_id=team_id)

        # Slice the rollout into distinct sequences.
        sequences = []
        for s in range(n_sequences):
            start = s * self.seq_len
            end = start + self.seq_len
            logger.debug('Slicing sequence {} from [{}:{}]'.format(s, start, end))

            sliced_states = {key: None for key in Policy.INPUT_KEYS}
            for key in sliced_states:
                sliced_states[key] = states[key][start:end, :].detach()

            sequence = Sequence(
                game_id=data['game_id'],
                states=sliced_states,
                actions=actions[start:end, :].detach(),
                masks=masks[start:end, :].detach(),
                rewards=rewards[start:end, :],
                discounted_rewards=discounted_rewards[start:end],
                weight_version=data['weight_version'],
                team_id=data['team_id'],
            )
            # Normalize the processed discounted rewards
            sequence.calculate_normalized_rewards(mean=self.running_mean[team_id],
                                                  std=self.running_std[team_id])

            # Finally, get the probabilties with the current policy.
            # sequence.calculate_old_probs(policy=self.policy_old)
            sequences.append(sequence)
        return sequences

    @staticmethod
    def list_of_dicts_to_dict_of_lists(x):
        return {k: torch.stack([d[k] for d in x]) for k in x[0]}

    def run(self):
        assert self.seq_per_epoch >= self.batch_size
        assert self.seq_per_epoch % self.batch_size == 0

        for it in range(self.iteration_start, self.iterations):
            logger.info('iteration {}/{}'.format(it, self.iterations))

            # First grab a bunch of experiences
            experiences = []
            subrewards = []
            rollout_lens = []
            weight_ages = []
            canvas = None  # Just save only the last canvas.
            while len(experiences) < self.seq_per_epoch:  # TODO(tzaman): with this approach, we often grab too many sequences!
                # logger.debug(' adding experience @{}/{}'.format(len(experiences), self.seq_per_epoch))

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
            advantages = []
            for ep in range(self.epochs):
                logger.info(' epoch {}/{}'.format(ep + 1, self.epochs))
                self.mq.process_data_events()

                # Shuffle the list of experience chunks.
                random.shuffle(experiences)

                # Divide into batches
                batches = [experiences[ib:ib + self.batch_size] for ib in range(0, len(experiences), self.batch_size)]
                for batch in batches:
                    loss_d, entropy_d, advantage = self.train(experiences=batch)
                    losses.append(loss_d)
                    entropies.append(entropy_d)
                    advantages.append(advantage)
            
            # Set the new policy as the old one.
            self.policy_old.load_state_dict(self.policy.state_dict())

            losses = self.list_of_dicts_to_dict_of_lists(losses)
            loss = losses['loss'].mean()

            advantages = torch.stack(advantages)
            advantage = advantages.mean()

            entropies = self.list_of_dicts_to_dict_of_lists(entropies)
            entropy = torch.stack(list(entropies.values())).sum(dim=0).mean()

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
                'loss/advantage': losses['advantage_loss'].mean(),
                'entropy': entropy,
                'advantage': advantage,
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

            for k, v in reward_dict.items():
                metrics['reward_per_sec/{}'.format(k)] = v

            logger.info('steps_per_s={:.2f}, avg_weight_age={:.1f}, reward_per_sec={:.4f}, loss={:.4f}, entropy={:.3f}, advantage={:.3f}'.format(
                steps_per_s, float(avg_weight_age), reward_per_sec, float(loss), float(entropy), float(advantage)))

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
        logger.debug('train(experiences={})'.format(experiences))

        # Batch together all experiences.
        vec_mh_rewards_norm = torch.cat([e.vec_mh_rewards_norm for e in experiences])
        # The action mask contains the mask of the selected actions
        vec_action_mask = torch.cat([e.vec_action_mask for e in experiences])
        # The head mask contains the mask of the relevant heads, where a selection has taken place,
        # and includes only valid possible selections from those heads.
        head_mask = torch.stack([e.vec_head_mask for e in experiences])  # [b, s, 59]
        # vec_old_probs = torch.cat([e.vec_old_probs for e in experiences])

        states = {key: [] for key in Policy.INPUT_KEYS}
        for e in experiences:
            for key in e.states:
                states[key].append(e.states[key])
        states = {key: torch.stack(states[key]) for key in states}

        # Notice there is no notion of loss masking here, this is unnessecary as we only work
        # use selected probabilties. E.g. when things were padded, nothing was selected, so no data.
        head_logits_dict, values, _ = self.policy(**states, hidden=None)  # -> {heads: tensors}

        # Advantage
        advantage = values - torch.from_numpy(e.discounted_rewards)

        mask_dict = Policy.unpack_heads(head_mask)

        # Perform a masked softmax
        head_log_prob_dict = {}
        for key in head_logits_dict:
            head_log_prob_dict[key] = Policy.masked_softmax(logits=head_logits_dict[key], mask=mask_dict[key])

        vec_log_probs_all = Policy.flatten_head(inputs=head_log_prob_dict).view(-1)

        # Now mask the probs by the selection
        vec_selected_log_probs = torch.masked_select(input=vec_log_probs_all, mask=vec_action_mask)

        # # PPO
        # # Probability ratio
        # rt = vec_selected_probs / (vec_old_probs + eps)

        # # PPO Objective
        # surr1 = rt * vec_mh_rewards_norm
        # surr2 = torch.clamp(rt, min=1.0 - self.e_clip, max=1.0 + self.e_clip) * vec_mh_rewards_norm
        # policy_loss = -torch.min(surr1, surr2).mean()  # This way, a positive reward will always lead to a negative loss

        # VPO
        policy_loss = -vec_selected_log_probs * vec_mh_rewards_norm
        policy_loss = policy_loss.mean()

        # # Check the entropy per head.
        entropies = {}
        for key in head_logits_dict:
            logp = head_log_prob_dict[key].clone()
            logp[~mask_dict[key]] = 0.
            p = torch.exp(logp).clone()
            p[~mask_dict[key]] = 0.
            e = p * logp
            e[~mask_dict[key]] = 0  # Zero out any non-used actions
            e = e.sum(dim=2)
            if e.size(0) == 0:
                # When no action of this kind was chosen.
                e = torch.zeros([])
            entropies[key] = -e.mean()

        if self.entropy_coef > 0:
            entropy = torch.stack(list(entropies.values())).sum()
            entropy_loss = -self.entropy_coef * entropy
        else:
            entropy_loss = torch.tensor(0.)

        if self.vf_coef > 0:
            advantage_loss = self.vf_coef * (advantage.pow(2)).mean()
        else:
            advantage_loss = torch.tensor(0.)

        loss = policy_loss + entropy_loss + advantage_loss

        if torch.isnan(loss):
            raise ValueError('loss={}, policy_loss={}, entropy_loss={}, advantage_loss={}'.format(
                loss, policy_loss, entropy_loss, advantage_loss))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.MAX_GRAD_NORM)
        self.optimizer.step()
        losses = {
            'loss': loss,
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
            'advantage_loss': advantage_loss,
        }
        return losses, entropies, advantage.mean()


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


def main(rmq_host, rmq_port, epochs, seq_per_epoch, batch_size, seq_len, learning_rate,
         pretrained_model, mq_prefetch_count, log_dir, entropy_coef, vf_coef, run_local):
    logger.info('main(rmq_host={}, rmq_port={}, epochs={} seq_per_epoch={}, batch_size={},'
                ' seq_len={} learning_rate={}, pretrained_model={}, mq_prefetch_count={}, entropy_coef={}, vf_coef={})'.format(
        rmq_host, rmq_port, epochs, seq_per_epoch, batch_size, seq_len, learning_rate, pretrained_model, mq_prefetch_count,
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
        seq_per_epoch=seq_per_epoch,
        batch_size=batch_size,
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
    parser.add_argument("--seq-per-epoch", type=int, help="amount of sequences per epoch", default=16)
    parser.add_argument("--batch-size", type=int, help="batch size", default=4)
    parser.add_argument("--seq-len", type=int, help="sequence length (as one sample in a minibatch)", default=256)
    parser.add_argument("--learning-rate", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--entropy-coef", type=float, help="entropy coef (as proportional addition to the loss)", default=0.01)
    parser.add_argument("--vf-coef", type=float, help="value fn coef (as proportional addition to the loss)", default=0.5)
    parser.add_argument("--pretrained-model", type=str, help="pretrained model file within gcs bucket", default=None)
    parser.add_argument("--mq-prefetch-count", type=int,
                        help="amount of experience messages to prefetch from mq", default=4)
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
            seq_per_epoch=args.seq_per_epoch,
            batch_size=args.batch_size,
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
