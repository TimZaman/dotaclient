from collections import Counter
import argparse
import io
import logging
import os
import pickle
import time

from google.cloud import storage
from tensorboardX import SummaryWriter
import numpy as np
import pika
import torch
import torch.distributed as dist

from policy import Policy
from distributed import DistributedDataParallelSparseParamCPU

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.manual_seed(7)

eps = np.finfo(np.float32).eps.item()

LEARNING_RATE = 1e-4

client = storage.Client()
bucket = client.get_bucket('dotaservice')

USE_CHECKPOINTS = True
MODEL_FILENAME_FMT = "model_%09d.pt"

START_EPISODE = 0
PRETRAINED_MODEL = None

# START_EPISODE = 647
# PRETRAINED_MODEL = 'runs/Dec30_22-46-22_optimizer-55f6d8fd9c-2c788/' + MODEL_FILENAME_FMT % START_EPISODE
# model_blob = bucket.get_blob(PRETRAINED_MODEL)
# PRETRAINED_MODEL = '/tmp/mdl.pt'
# model_blob.download_to_filename(PRETRAINED_MODEL)


class Experience:
    def __init__(self, game_id, states, actions, rewards, weight_version):
        self.game_id = game_id
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.weight_version = weight_version


class DotaOptimizer:

    EXPERIENCE_QUEUE_NAME = 'experience'
    MODEL_EXCHANGE_NAME = 'model'

    def __init__(self, rmq_host, rmq_port, batch_size):
        super().__init__()
        self.rmq_host = rmq_host
        self.rmq_port = rmq_port
        self.batch_size = batch_size

        self.policy = Policy()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.policy = DistributedDataParallelSparseParamCPU(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.time_last_step = time.time()

        pretrained_model = PRETRAINED_MODEL
        if pretrained_model is not None:
            self.policy.load_state_dict(torch.load(pretrained_model), strict=False)

        self.episode = START_EPISODE

        self.writer = SummaryWriter()
        logger.info('Checkpointing to: {}'.format(self.log_dir))

        # self.rmq_connection = pika.BlockingConnection(pika.ConnectionParameters(
        #     host=rmq_host,
        #     port=rmq_port,
        #     heartbeat=300,
        #     ))
        # self.experience_channel = self.rmq_connection.channel()
        # self.experience_channel.basic_qos(prefetch_count=self.batch_size)
        # self.experience_channel.queue_declare(queue=self.EXPERIENCE_QUEUE_NAME)

        # self.model_exchange = self.rmq_connection.channel()
        # self.model_exchange.exchange_declare(
        #     exchange=self.MODEL_EXCHANGE_NAME,
        #     exchange_type='x-recent-history',
        #     arguments={'x-recent-history-length':1, 'x-recent-history-no-store':True},
        #     )

    @property
    def events_filename(self):
        return self.writer.file_writer.event_writer._ev_writer._file_name

    @property
    def log_dir(self):
        return self.writer.file_writer.get_logdir()

    @staticmethod
    def discount_rewards(rewards, gamma=0.99):
        R = 0
        discounted_rewards = []
        for r in rewards[::-1]:
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        return discounted_rewards

    def finish_episode(self, rewards, log_probs):
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        loss = []
        for log_probs, reward in zip(log_probs, rewards):
            loss.append(-log_probs * reward)

        self.optimizer.zero_grad()
        loss = torch.cat(loss).mean()
        loss.backward()
        self.optimizer.step()

        return loss

    def process_rollout(self, states, actions):
        hidden = None
        all_rewards = []
        log_prob_sum = []

        # Loop over each step.
        for policy_input, action_dict in zip(states, actions):
            head_prob_dict, hidden = self.policy(**policy_input, hidden=hidden)

            action_probs = self.policy.module.action_probs(  #HACK!
                head_prob_dict=head_prob_dict,
                action_dict=action_dict,
            )
            # all_action_probs.append(action_probs)
            log_prob_sum.append(sum([ap.log_prob for _, ap in action_probs.items()]))
        return log_prob_sum

    def run(self):
        while True:
            rmq_connection = pika.BlockingConnection(pika.ConnectionParameters(
                host=self.rmq_host,
                port=self.rmq_port,
                heartbeat=0,
                ))
            experience_channel = rmq_connection.channel()
            experience_channel.basic_qos(prefetch_count=self.batch_size)
            experience_channel.queue_declare(queue=self.EXPERIENCE_QUEUE_NAME)
            experiences = []
            for _ in range(self.batch_size):
                method_frame, properties, body = next(experience_channel.consume(
                    queue=self.EXPERIENCE_QUEUE_NAME,
                    no_ack=True,
                    ))
                data = pickle.loads(body)
                experience = Experience(
                    game_id=data['game_id'],
                    states=data['states'],
                    actions=data['actions'],
                    rewards=data['rewards'],
                    weight_version=data['weight_version'],
                    )
                experiences.append(experience)
            self.step(experiences=experiences)
            rmq_connection.close()

    def step(self, experiences):
        logger.info('::step episode={}'.format(self.episode))
        # Get item form queue
        all_reward_sums = []
        all_discounted_rewards = []
        all_logprobs = []
        all_rewards = []
        all_weight_ages = []

        # Loop over each experience
        for experience in experiences:
            # self.rmq_connection.process_data_events()  # Process RMQ heartbeats.
            log_prob_sum = self.process_rollout(
                states=experience.states,
                actions=experience.actions,
            )

            all_rewards.append(experience.rewards)
            reward_sums = [sum(r.values()) for r in experience.rewards]
            discounted_rewards = self.discount_rewards(reward_sums)

            all_reward_sums.append(sum(reward_sums))

            all_discounted_rewards.extend(discounted_rewards)
            all_logprobs.extend(log_prob_sum)
            all_weight_ages.append(self.episode - experience.weight_version)

        n_steps = len(all_discounted_rewards)

        loss = self.finish_episode(rewards=all_discounted_rewards, log_probs=all_logprobs)

        self.episode += 1

        steps_per_s = n_steps / (time.time() - self.time_last_step)
        self.time_last_step = time.time()

        avg_weight_age = sum(all_weight_ages) / self.batch_size

        logger.info('loss={:.4f}, steps_per_s={:.2f}, avg_weight_age={:.2f}'.format(loss, steps_per_s, avg_weight_age))

        reward_counter = Counter()
        for b in all_rewards:  # Jobs in a batch.
            for s in b:  # Steps in a batch.
                reward_counter.update(s)
        reward_counter = dict(reward_counter)

        reward_sum = sum(reward_counter.values())
        mean_reward = reward_sum / self.batch_size

        logger.info('mean_reward={}'.format(mean_reward))

        if USE_CHECKPOINTS:

            metrics = {
                'steps per s': steps_per_s,
                'avg weight age': avg_weight_age,
                'loss': loss,
                'mean_reward': mean_reward,
            }
            for k, v in reward_counter.items():
                metrics['reward_{}'.format(k)] = v / self.batch_size

            # Reduce all the metrics
            metrics_t = torch.tensor(list(metrics.values()), dtype=torch.float32)
            dist.all_reduce(metrics_t, op=dist.ReduceOp.SUM)
            metrics_t /= dist.get_world_size()

            for name, metric in zip(metrics.keys(), metrics_t):
                if name == 'steps per s':
                    # The speed is always the sum. TODO(tzaman): improve
                    metric *= dist.get_world_size()
                self.writer.add_scalar(name, metric, self.episode)

            # Upload events to GCS
            blob = bucket.blob(self.events_filename)
            blob.upload_from_filename(filename=self.events_filename)

            self.upload_model()

    def upload_model(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            # Only rank 0 uploads the model.
            return

        filename = MODEL_FILENAME_FMT % self.episode
        rel_path = os.path.join(self.log_dir, filename)

        # Serialize the model.
        buffer = io.BytesIO()
        state_dict = self.policy.state_dict()
        # Potentially remove the "module." suffix induced by DataParallel wrapper.
        for key in list(state_dict.keys()):
            if key[:7] == 'module.':
                val = state_dict[key]
                del state_dict[key]
                state_dict[key[7:]] = val
        torch.save(state_dict, buffer)

        # Write model to file.
        with open(rel_path,'wb') as f:
            f.write(buffer.read())

        # Send to exchange.
        rmq_connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=self.rmq_host,
            port=self.rmq_port,
            heartbeat=0,
        ))
        model_exchange = rmq_connection.channel()
        model_exchange.exchange_declare(
            exchange=self.MODEL_EXCHANGE_NAME,
            exchange_type='x-recent-history',
            arguments={'x-recent-history-length':1, 'x-recent-history-no-store':True},
            )
        model_exchange.basic_publish(
            exchange=self.MODEL_EXCHANGE_NAME,
            routing_key='',
            body=buffer.getbuffer(),
            properties=pika.BasicProperties(headers={'version': self.episode}),
            )
        rmq_connection.close()

        # Upload to GCP.
        blob = bucket.blob(rel_path)
        blob.upload_from_filename(filename=rel_path)  # Model


def init_distribution():
    logger.info('init_distribution')
    assert 'WORLD_SIZE' in os.environ
    world_size = int(os.environ['WORLD_SIZE'])
    if world_size < 2:
        return

    assert 'MASTER_ADDR' in os.environ
    assert 'MASTER_PORT' in os.environ

    # For the rank, we depend on the hostname's trailing ordinal index (StatefulSet)
    hostname = os.environ['HOSTNAME']
    rank = int(hostname.split('-')[-1])

    if rank != 0:
        USE_CHECKPOINTS = False

    logger.info('hostname={}, rank={}, world_size={}'.format(hostname, rank, world_size))
    logger.info('MASTER_ADDR={}, MASTER_PORT={}'.format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))

    torch.distributed.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    logger.info("Distribution initialized.")


def main(rmq_host, rmq_port, batch_size):
    logger.info('main(rmq_host={}, rmq_port={}, batch_size={})'.format(rmq_host, rmq_port, batch_size))
    if torch.distributed.is_available():
        init_distribution()
    else:
        logger.info('distribution unavailable')

    dota_optimizer = DotaOptimizer(rmq_host=rmq_host, rmq_port=rmq_port, batch_size=batch_size)

    # Upload initial model.
    dota_optimizer.upload_model()

    dota_optimizer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", type=str, help="mq ip", default='127.0.0.1')
    parser.add_argument("--port", type=int, help="mq port", default=5672)
    parser.add_argument("--batch-size", type=int, help="batch size", default=8)
    args = parser.parse_args()

    try:
        main(rmq_host=args.ip, rmq_port=args.port, batch_size=args.batch_size)
    except KeyboardInterrupt:
        pass
