from collections import Counter
from pprint import pprint, pformat
import argparse
import io
import logging
import os
import pickle
import pika
import time

from google.cloud import storage
from tensorboardX import SummaryWriter
import numpy as np
import torch

from policy import Policy


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

torch.manual_seed(7)

eps = np.finfo(np.float32).eps.item()

LEARNING_RATE = 1e-4
MIN_BATCH_SIZE = 4

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


class DotaOptimizer():

    EXPERIENCE_QUEUE_NAME = 'experience'
    MODEL_EXCHANGE_NAME = 'model'

    def __init__(self, rmq_host, rmq_port):
        super().__init__()
        self.rmq_host = rmq_host
        self.rmq_port = rmq_port
        self.policy = Policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.time_last_step = time.time()

        pretrained_model = PRETRAINED_MODEL
        if pretrained_model:
            self.policy.load_state_dict(torch.load(pretrained_model), strict=False)

        self.episode = START_EPISODE

        self.writer = SummaryWriter()
        logger.info('Checkpointing to: {}'.format(self.log_dir))

        self.rmq_connection = pika.BlockingConnection(pika.ConnectionParameters(host=rmq_host, port=rmq_port))
        self.experience_channel = self.rmq_connection.channel()
        self.experience_channel.queue_declare(queue=self.EXPERIENCE_QUEUE_NAME)
        self.model_exchange = self.rmq_connection.channel()
        self.model_exchange.exchange_declare(
            exchange=self.MODEL_EXCHANGE_NAME,
            exchange_type='x-recent-history',
            arguments={'x-recent-history-length':1, 'x-recent-history-no-store':True},
            )

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
        # print('::finish_episode')
        # print('rewards=\n{}'.format(rewards))
        # print('log_probs=\n{}'.format(log_probs))

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        loss = []
        for log_probs, reward in zip(log_probs, rewards):
            loss.append(-log_probs * reward)

        self.optimizer.zero_grad()
        loss = torch.cat(loss).mean()
        loss.backward()
        self.policy.maybe_average_gradients()
        self.optimizer.step()

        return loss

    def process_rollout(self, states, actions):
        # print('::process_rollout')
        hidden = None
        # all_action_probs = []
        all_rewards = []
        log_prob_sum = []

        # Loop over each step.
        for policy_input, action_dict in zip(states, actions):
            head_prob_dict, hidden = self.policy(**policy_input, hidden=hidden)
            action_probs = self.policy.action_probs(
                head_prob_dict=head_prob_dict,
                action_dict=action_dict,
            )
            # all_action_probs.append(action_probs)
            log_prob_sum.append(sum([ap.log_prob for _, ap in action_probs.items()]))
        return log_prob_sum

    def run(self):
        while True:
            experiences = []
            for _ in range(MIN_BATCH_SIZE):
                method_frame, properties, body = next(self.experience_channel.consume(
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

    def step(self, experiences):
        logger.info('::step episode={}'.format(self.episode))
        # Get item form queue
        all_reward_sums = []
        all_discounted_rewards = []
        all_logprobs = []
        all_rewards = []
        all_weight_ages = []

        batch_size = len(experiences)
        # Loop over each experience
        for experience in experiences:
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

        steps_per_s = n_steps / (time.time() - self.time_last_step)
        self.time_last_step = time.time()

        avg_weight_age = float(sum(all_weight_ages)) / batch_size

        logger.info('loss={:.4f}, steps_per_s={:.2f}, avg_weight_age={:.2f}'.format(loss, steps_per_s, avg_weight_age))

        reward_counter = Counter()
        for b in all_rewards:  # Jobs in a batch.
            for s in b:  # Steps in a batch.
                reward_counter.update(s)
        reward_counter = dict(reward_counter)

        reward_sum = sum(reward_counter.values())
        mean_reward = reward_sum / batch_size

        logger.info('mean_reward={}'.format(mean_reward))

        if USE_CHECKPOINTS:
            self.writer.add_scalar('batch_size', batch_size, self.episode)
            self.writer.add_scalar('steps per s', steps_per_s, self.episode)
            self.writer.add_scalar('avg weight age', avg_weight_age, self.episode)
            self.writer.add_scalar('loss', loss, self.episode)
            self.writer.add_scalar('mean_reward', mean_reward, self.episode)
            for k, v in reward_counter.items():
                self.writer.add_scalar('reward_{}'.format(k), v / batch_size, self.episode)
            # Upload events to GCS
            blob = bucket.blob(self.events_filename)
            blob.upload_from_filename(filename=self.events_filename)

            self.upload_model()

        self.episode += 1


    def upload_model(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            # Only rank 0 uploads the model.
            return

        filename = MODEL_FILENAME_FMT % self.episode
        rel_path = os.path.join(self.log_dir, filename)

        # Serialize the model.
        buffer = io.BytesIO()
        torch.save(self.policy.state_dict(), buffer)

        # Write model to file.
        with open(rel_path,'wb') as f:
            f.write(buffer.read())

        # Send to exchange.
        self.model_exchange.basic_publish(
            exchange=self.MODEL_EXCHANGE_NAME,
            routing_key='',
            body=buffer.getbuffer(),
            properties=pika.BasicProperties(headers={'version': self.episode}),
            )

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
    
    print('hostname={}, rank={}, world_size={}'.format(hostname, rank, world_size))
    print('MASTER_ADDR={}, MASTER_PORT={}'.format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))

    torch.distributed.init_process_group(backend='gloo', rank=rank, world_size=world_size)


def main(rmq_host, rmq_port):
    print('main(rmq_host={}, rmq_port={})'.format(rmq_host, rmq_port))
    if torch.distributed.is_available():
        init_distribution()
    else:
        logger.info('distribution unavailable')

    dota_optimizer = DotaOptimizer(rmq_host, rmq_port)

    # Upload initial model.
    dota_optimizer.upload_model()

    dota_optimizer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", type=str, help="mq ip", default='127.0.0.1')
    parser.add_argument("--port", type=str, help="mq port", default=5672)
    args = parser.parse_args()

    try:
        main(rmq_host=args.ip, rmq_port=args.port)
    except KeyboardInterrupt:
        pass
