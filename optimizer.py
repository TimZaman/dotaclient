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
from policy import RndModel
from distributed import DistributedDataParallelSparseParamCPU

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.manual_seed(7)

eps = np.finfo(np.float32).eps.item()


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def is_master():
    if is_distributed():
        return torch.distributed.get_rank() == 0
    else:
        return True


class MessageQueue:
    EXPERIENCE_QUEUE_NAME = 'experience'
    MODEL_EXCHANGE_NAME = 'model'
    MAX_RETRIES = 10

    def __init__(self, host, port, prefetch_count):
        self._params = pika.ConnectionParameters(
            host=host,
            port=port,
            heartbeat=0,
        )
        self.prefetch_count = prefetch_count
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
            self._model_exchange = self._conn.channel()
            self._model_exchange.exchange_declare(
                exchange=self.MODEL_EXCHANGE_NAME,
                exchange_type='x-recent-history',
                arguments={'x-recent-history-length': 1},
            )

    def process_data_events(self):
        # Seems useless..
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
            no_ack=True,
            # no_ack=False,
        ))
        return method, properties, body

    def consume_xp(self):
        try:
            return self._consume_xp()
        except (pika.exceptions.ConnectionClosed, pika.exceptions.ChannelClosed):
            logger.error('reconnecting to queue')
            self.connect()
            return self._consume_xp()

    # def ack_xp(self, tags):
    #     try:
    #         for tag in tags:
    #             self._xp_channel.basic_ack(delivery_tag=tag)
    #     except (pika.exceptions.ConnectionClosed, pika.exceptions.ChannelClosed):
    #         logger.error('ack failed')

    def close(self):
        if self._conn and self._conn.is_open:
            logger.info('closing queue connection')
            self._conn.close()


class Experience:
    def __init__(self, game_id, states, actions, rewards, weight_version):
        self.game_id = game_id
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.weight_version = weight_version


class DotaOptimizer:

    MODEL_FILENAME_FMT = "model_%09d.pt"
    BUCKET_NAME = 'dotaservice'

    def __init__(self, rmq_host, rmq_port, batch_size, learning_rate, checkpoint, pretrained_model):
        super().__init__()
        self.rmq_host = rmq_host
        self.rmq_port = rmq_port
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint = checkpoint
        self.episode = 0

        self.policy_base = Policy()

        if self.checkpoint:
            # TODO(tzaman): Set logdir ourselves?
            self.writer = SummaryWriter()
            logger.info('Checkpointing to: {}'.format(self.log_dir))
            client = storage.Client()
            self.bucket = client.get_bucket(self.BUCKET_NAME)

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

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.time_last_step = time.time()

        self.rnd_fixed = RndModel(requires_grad=False)
        self.rnd = RndModel(requires_grad=True)
        self.rnd_lr = 1e-3
        self.rnd_optimizer = torch.optim.Adam(self.rnd.parameters(), lr=self.rnd_lr)


        self.mq = MessageQueue(host=self.rmq_host, port=self.rmq_port,
                               prefetch_count=self.batch_size)
        self.mq.connect()


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

    def process_rnd(self, states):
        rnd_loss = []
        rnd_rewards = []
        for policy_input in states:
            of = self.rnd_fixed(**policy_input)
            o = self.rnd(**policy_input)
            loss = torch.nn.functional.mse_loss(of, o)
            rnd_loss.append(loss)  # (~0.001 range)
            rnd_rewards.append(loss/1000.)
        return rnd_loss, rnd_rewards

    def rnd_step(self, rnd_losses):
        self.rnd_optimizer.zero_grad()
        rnd_loss = torch.stack(rnd_losses).mean()
        rnd_loss.backward()
        self.rnd_optimizer.step()
        return rnd_loss

    def process_rollout(self, states, actions):
        hidden = None
        all_rewards = []
        log_prob_sum = []

        # Loop over each step.
        for policy_input, action_dict in zip(states, actions):
            head_prob_dict, hidden = self.policy(**policy_input, hidden=hidden)

            action_probs = self.policy_base.action_probs(  # HACK!
                head_prob_dict=head_prob_dict,
                action_dict=action_dict,
            )
            log_prob_sum.append(sum([ap.log_prob for _, ap in action_probs.items()]))
            
        return log_prob_sum

    def run(self):
        while True:
            experiences = []
            # delivered_tags = []
            for _ in range(self.batch_size):
                method, properties, body = self.mq.consume_xp()
                # delivered_tags.append(method.delivery_tag)
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
            # self.mq.ack_xp(tags=delivered_tags)

    def step(self, experiences):
        logger.info('::step episode={}'.format(self.episode))
        # Get item form queue
        all_reward_sums = []
        all_discounted_rewards = []
        all_logprobs = []
        all_rewards = []
        all_weight_ages = []
        all_rnd_loss = []

        # Loop over each experience
        for experience in experiences:
            self.mq.process_data_events()  # Seems useless.
            rnd_loss, rnd_rewards = self.process_rnd(experience.states)
            all_rnd_loss.extend(rnd_loss)
            # Add to rewards somehow.
            for r, rnd_reward in zip(experience.rewards, rnd_rewards):
                r['rnd'] = rnd_reward

            # self.mq.process_events()
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

        rnd_loss = self.rnd_step(all_rnd_loss)

        n_steps = len(all_discounted_rewards)

        loss = self.finish_episode(rewards=all_discounted_rewards, log_probs=all_logprobs)

        self.episode += 1

        steps_per_s = n_steps / (time.time() - self.time_last_step)
        self.time_last_step = time.time()

        avg_weight_age = sum(all_weight_ages) / self.batch_size

        reward_counter = Counter()
        for b in all_rewards:  # Jobs in a batch.
            for s in b:  # Steps in a batch.
                reward_counter.update(s)
        reward_counter = dict(reward_counter)

        reward_sum = sum(reward_counter.values())
        mean_reward = reward_sum / self.batch_size

        logger.info('steps_per_s={:.2f}, avg_weight_age={:.2f}, mean_reward={:.2f}, loss={:.4f}, rnd_loss={:.5f}'.format(
            steps_per_s, avg_weight_age, mean_reward, loss, rnd_loss))

        speed_key = 'steps per s'
        metrics = {
            speed_key: steps_per_s,
            'avg weight age': avg_weight_age,
            'mean_reward': mean_reward,
            'loss': loss,
            'rnd_loss': rnd_loss,
        }
        for k, v in reward_counter.items():
            metrics['reward_{}'.format(k)] = v / self.batch_size

        # Reduce all the metrics
        metrics_t = torch.tensor(list(metrics.values()), dtype=torch.float32)

        if is_distributed():
            dist.all_reduce(metrics_t, op=dist.ReduceOp.SUM)
            metrics_t /= dist.get_world_size()

        metrics_d = dict(zip(metrics.keys(), metrics_t))

        if is_distributed():
            # Speed is always the sum.
            metrics_d[speed_key] *= dist.get_world_size()

        if self.checkpoint:
            # Write metrics to events file.
            for name, metric in metrics_d.items():
                self.writer.add_scalar(name, metric, self.episode)

            # Upload events to GCS
            blob = self.bucket.blob(self.events_filename)
            blob.upload_from_filename(filename=self.events_filename)

            self.upload_model()

    def upload_model(self):
        if not is_master():
            # Only rank 0 uploads the model.
            return

        filename = self.MODEL_FILENAME_FMT % self.episode
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
        self.mq.publish_model(msg=state_dict_b, hdr={'version': self.episode})

        # Upload to GCP.
        blob = self.bucket.blob(rel_path)
        blob.upload_from_string(data=state_dict_b)  # Model


def init_distribution(backend='gloo'):
    logger.info('init_distribution')
    assert 'WORLD_SIZE' in os.environ
    if int(os.environ['WORLD_SIZE']) < 2:
        return
    torch.distributed.init_process_group(backend=backend)
    logger.info("Distribution initialized.")


def main(rmq_host, rmq_port, batch_size, learning_rate, pretrained_model):
    logger.info('main(rmq_host={}, rmq_port={}, batch_size={})'.format(rmq_host, rmq_port, batch_size))
 
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
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint=checkpoint,
        pretrained_model=pretrained_model,
    )

    # Upload initial model.
    dota_optimizer.upload_model()

    dota_optimizer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", type=str, help="mq ip", default='127.0.0.1')
    parser.add_argument("--port", type=int, help="mq port", default=5672)
    parser.add_argument("--batch-size", type=int, help="batch size", default=8)
    parser.add_argument("--learning-rate", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--pretrained-model", type=str, help="pretrained model file within gcs bucket", default=None)
    args = parser.parse_args()

    try:
        main(
            rmq_host=args.ip,
            rmq_port=args.port,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            pretrained_model=args.pretrained_model,
        )
    except KeyboardInterrupt:
        pass
