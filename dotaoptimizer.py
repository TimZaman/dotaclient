from collections import Counter
from pprint import pprint, pformat
import asyncio
import logging
import os
import pickle
import time

from google.cloud import storage
from grpclib.server import Server
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.optim as optim

from policy import Policy
from protos.DotaOptimizer_grpc import DotaOptimizerBase
from protos.DotaOptimizer_grpc import DotaOptimizerBase
from protos.DotaOptimizer_pb2 import Empty2
from protos.DotaOptimizer_pb2 import Weights

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

eps = np.finfo(np.float32).eps.item()

LEARNING_RATE = 1e-4
MIN_BATCH_SIZE = 4

client = storage.Client()
bucket = client.get_bucket('dotaservice')



START_EPISODE = 31
USE_CHECKPOINTS = True
MODEL_FILENAME_FMT = "model_%09d.pt"

# PRETRAINED_MODEL = None
PRETRAINED_MODEL = 'runs/Dec24_15-44-04_Tims-MacBook-Pro.local/' + MODEL_FILENAME_FMT % START_EPISODE
model_blob = bucket.get_blob(PRETRAINED_MODEL)
PRETRAINED_MODEL = '/tmp/mdl.pt'
model_blob.download_to_filename(PRETRAINED_MODEL)



class Experience:
    def __init__(self, game_id, states, actions, rewards):
        self.game_id = game_id
        self.states = states
        self.actions = actions
        self.rewards = rewards


class DotaOptimizer(DotaOptimizerBase):

    def __init__(self):
        super().__init__()
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.experience_queue = asyncio.Queue(loop=asyncio.get_event_loop())
        self.time_last_step = time.time()

        pretrained_model = PRETRAINED_MODEL
        if pretrained_model:
            self.policy.load_state_dict(torch.load(pretrained_model), strict=False)

        self.episode = START_EPISODE

        if USE_CHECKPOINTS:
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

    async def step(self):
        print('::step')
        # Get item form queue
        all_reward_sums = []
        all_discounted_rewards = []
        all_logprobs = []
        all_rewards = []

        # Loop over each experience
        batch_size = None
        while self.experience_queue.qsize() > 0:
            experience = await self.experience_queue.get()
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
        batch_size = len(all_reward_sums)

        loss = self.finish_episode(rewards=all_discounted_rewards, log_probs=all_logprobs)

        steps_per_s = n_steps / (time.time() - self.time_last_step)
        self.time_last_step = time.time()

        print('loss=', loss)
        print('steps_per_s=', steps_per_s)

        reward_counter = Counter()
        for b in all_rewards: # Jobs in a batch.
            for s in b: # Steps in a batch.
                reward_counter.update(s)
        reward_counter = dict(reward_counter)

        reward_sum = sum(reward_counter.values())
        mean_reward = reward_sum / batch_size

        print('mean_reward=', mean_reward)

        if USE_CHECKPOINTS:
            # TODO(tzaman): should we write a model snapshot at episode 0?

            self.writer.add_scalar('steps per s', steps_per_s, self.episode)
            self.writer.add_scalar('loss', loss, self.episode)
            self.writer.add_scalar('mean_reward', mean_reward, self.episode)
            for k, v in reward_counter.items():
                self.writer.add_scalar('reward_{}'.format(k), v / batch_size, self.episode)
            filename = MODEL_FILENAME_FMT % self.episode
            rel_path = os.path.join(self.log_dir, filename)
            torch.save(self.policy.state_dict(), rel_path)

            # Upload to GCP.
            blob = bucket.blob(rel_path)
            blob.upload_from_filename(filename=rel_path)  # Model
            blob = bucket.blob(self.events_filename)
            blob.upload_from_filename(filename=self.events_filename)  # Events file

        self.episode += 1
        

    async def Rollout(self, stream):
        # print('::Rollout')
        request = await stream.recv_message()

        states = pickle.loads(request.states)
        actions = pickle.loads(request.actions)
        rewards = pickle.loads(request.rewards)

        experience = Experience(game_id=request.game_id, states=states, actions=actions, rewards=rewards)

        await self.experience_queue.put(experience)

        if self.experience_queue.qsize() >= MIN_BATCH_SIZE:
            await self.step()

        await stream.send_message(Empty2())  # TODO(tzaman): Respond earlier?

    async def GetWeights(self, stream):
        # print('::GetWeights')
        _ = await stream.recv_message()

        state_dict = self.policy.state_dict()
        state_dict_p = pickle.dumps(state_dict)

        await stream.send_message(Weights(data=state_dict_p))


async def serve(server, *, host='127.0.0.1', port=50051):
    await server.start(host, port)
    print('Serving on {}:{}'.format(host, port))
    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        server.close()
        await server.wait_closed()


async def main():
    server = Server([DotaOptimizer()], loop=asyncio.get_event_loop())
    await serve(server)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass