from collections import Counter
from pprint import pprint, pformat
import asyncio
import logging
import os
import pickle
import pika
import time

from google.cloud import storage
from grpclib.server import Server
import grpc
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.optim as optim

from policy import Policy

from protos.ModelService_pb2_grpc import ModelServiceStub
from protos.ModelService_pb2 import Weights


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

# USE_CHECKPOINTS = False
# MODEL_FILENAME_FMT = "model_%09d.pt"

START_EPISODE = 0
# PRETRAINED_MODEL = None

# START_EPISODE = 647
# PRETRAINED_MODEL = 'runs/Dec30_22-46-22_optimizer-55f6d8fd9c-2c788/' + MODEL_FILENAME_FMT % START_EPISODE
# model_blob = bucket.get_blob(PRETRAINED_MODEL)
# PRETRAINED_MODEL = '/tmp/mdl.pt'
# model_blob.download_to_filename(PRETRAINED_MODEL)


class Experience:
    def __init__(self, game_id, states, actions, rewards):
        self.game_id = game_id
        self.states = states
        self.actions = actions
        self.rewards = rewards


class DotaOptimizer():
    def __init__(self):
        super().__init__()
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.experience_queue = asyncio.Queue(loop=asyncio.get_event_loop())
        self.time_last_step = time.time()

        # pretrained_model = PRETRAINED_MODEL
        # if pretrained_model:
        #     self.policy.load_state_dict(torch.load(pretrained_model), strict=False)

        self.episode = START_EPISODE

        self.writer = SummaryWriter()
        logger.info('Checkpointing to: {}'.format(self.log_dir))


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
        connection = pika.BlockingConnection(pika.ConnectionParameters('127.0.0.1'))
        channel = connection.channel()
        while True:
            experiences = []
            for _ in range(MIN_BATCH_SIZE):
                method_frame, properties, body = next(channel.consume(queue='hello', no_ack=True))
                data = pickle.loads(body)
                experience = Experience(
                    game_id=data['game_id'], states=data['states'], actions=data['actions'], rewards=data['rewards'])
                experiences.append(experience)
            self.step(experiences=experiences)
        connection.close()

    def step(self, experiences):
        logger.info('::step episode={}'.format(self.episode))
        # Get item form queue
        all_reward_sums = []
        all_discounted_rewards = []
        all_logprobs = []
        all_rewards = []

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

        n_steps = len(all_discounted_rewards)

        loss = self.finish_episode(rewards=all_discounted_rewards, log_probs=all_logprobs)

        steps_per_s = n_steps / (time.time() - self.time_last_step)
        self.time_last_step = time.time()

        logger.info('loss={}'.format(loss))
        logger.info('steps_per_s={}'.format(steps_per_s))

        reward_counter = Counter()
        for b in all_rewards:  # Jobs in a batch.
            for s in b:  # Steps in a batch.
                reward_counter.update(s)
        reward_counter = dict(reward_counter)

        reward_sum = sum(reward_counter.values())
        mean_reward = reward_sum / batch_size

        logger.info('mean_reward={}'.format(mean_reward))

        # TODO(tzaman): should we write a model snapshot at episode 0?
        self.writer.add_scalar('batch_size', batch_size, self.episode)
        self.writer.add_scalar('steps per s', steps_per_s, self.episode)
        self.writer.add_scalar('loss', loss, self.episode)
        self.writer.add_scalar('mean_reward', mean_reward, self.episode)
        for k, v in reward_counter.items():
            self.writer.add_scalar('reward_{}'.format(k), v / batch_size, self.episode)

        # Upload events file.
        blob = bucket.blob(self.events_filename)
        blob.upload_from_filename(filename=self.events_filename)  # Events file

        # Upload model.
        self.upload_model()

        # filename = MODEL_FILENAME_FMT % self.episode
        # rel_path = os.path.join(self.log_dir, filename)
        # torch.save(self.policy.state_dict(), rel_path)
        # blob = bucket.blob(rel_path)
        # blob.upload_from_filename(filename=rel_path)  # Model


        self.episode += 1

    def upload_model(self):
        channel = grpc.insecure_channel('0.0.0.0:50052')
        stub = ModelServiceStub(channel)
        state_dict_pickled = pickle.dumps(self.policy.state_dict())
        _ = stub.PutWeights(Weights(
            version=self.episode,
            data=state_dict_pickled,
            log_dir=self.log_dir,
            ))


def init_distribution():
    logger.info('init_distribution')
    assert 'WORLD_SIZE' in os.environ
    world_size = int(os.environ['WORLD_SIZE'])
    if world_size == 1:
        return

    assert 'MASTER_ADDR' in os.environ
    assert 'MASTER_PORT' in os.environ

    # For the rank, we depend on the hostname's trailing ordinal index (StatefulSet)
    hostname = os.environ['HOSTNAME']
    rank = int(hostname.split('-')[-1])
    
    print('hostname={}, rank={}, world_size={}'.format(hostname, rank, world_size))
    print('MASTER_ADDR={}, MASTER_PORT={}'.format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))

    torch.distributed.init_process_group(backend='gloo', rank=rank, world_size=world_size)


def main():
    if torch.distributed.is_available():
        init_distribution()
    else:
        logger.info('distribution unavailable')

    dota_optimizer = DotaOptimizer()

    # Upload initial model.
    dota_optimizer.upload_model()

    dota_optimizer.run()




if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
