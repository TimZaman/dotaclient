from collections import Counter
from pprint import pprint, pformat
import asyncio
import logging
import os
import pickle
import time
import torch

from google.cloud import storage
from grpclib.server import Server
import numpy as np

from protos.ModelService_grpc import ModelServiceBase
from protos.ModelService_pb2 import Empty2
from protos.ModelService_pb2 import WeightQuery
from protos.ModelService_pb2 import Weights

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


client = storage.Client()
bucket = client.get_bucket('dotaservice')


# USE_CHECKPOINTS = False
MODEL_FILENAME_FMT = "model_%09d.pt"

# START_EPISODE = 0
# PRETRAINED_MODEL = None

# START_EPISODE = 647
# PRETRAINED_MODEL = 'runs/Dec30_22-46-22_optimizer-55f6d8fd9c-2c788/' + MODEL_FILENAME_FMT % START_EPISODE
# model_blob = bucket.get_blob(PRETRAINED_MODEL)
# PRETRAINED_MODEL = '/tmp/mdl.pt'
# model_blob.download_to_filename(PRETRAINED_MODEL)



class ModelService(ModelServiceBase):
    def __init__(self):
        super().__init__()
        self._latest_weights = None
        self.episode = None
        self.log_dir = '/tmp'

    @property
    def latest_weights(self):
        # If we don't have the latest weights, query the gcp folder.
        # bucket.list_blobs(prefix='runs/')
        return self._latest_weights

    async def PutWeights(self, stream):
        logger.info('::PutWeights')
        request = await stream.recv_message()
        # How do we get the snapshot directory?

        # TODO(tzaman): Make sure the weights are new.

        pickled_state_dict = request.data

        self._latest_weights = request.data
        self.episode = request.version

        # Save to GCS for persistance.
        filename = MODEL_FILENAME_FMT % self.episode
        rel_path = os.path.join(self.log_dir, filename)
        torch.save(pickle.loads(pickled_state_dict), rel_path)
        blob = bucket.blob(rel_path)
        blob.upload_from_filename(filename=rel_path)

        await stream.send_message(Empty2())

    async def GetWeights(self, stream):
        logger.info('::GetWeights')
        request = await stream.recv_message()
        version = request.version

        # Impossible to have a newer weight than the optimizer itself
        assert version <= self.episode

        if self.episode == version and version != 0:
            # TODO(tzaman): version -1 should just always update weights
            await stream.send_message(
                Weights(
                    status=Weights.Status.Value('UP_TO_DATE'),
                    version=self.episode,
                ))
            return

        pickled_state_dict = self.latest_weights

        await stream.send_message(
            Weights(
                status=Weights.Status.Value('OK'),
                version=self.episode,
                data=pickled_state_dict,
            ))


async def serve(server, host, port):
    logger.info('ModelService serving on {}:{}'.format(host, port))
    await server.start(host, port)
    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        server.close()
        await server.wait_closed()


async def main():

    server = Server([ModelService()], loop=asyncio.get_event_loop())
    await serve(server, host='', port=50052)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
