import logging

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.autograd import Variable
from torch.nn.modules import Module
import torch
import torch.distributed as dist
import torch.utils.hooks

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.manual_seed(7)

class DistributedDataParallelSparseParamCPU(Module):

    def __init__(self, module):
        logger.info('DistributedDataParallelSparseParamCPU::__init__')
        super(DistributedDataParallelSparseParamCPU, self).__init__()
        self.module = module
        self.sync_parameters()

        def allreduce_params():
            # print('allreduce_params needs_reduction={}'.format(self.needs_reduction))
            if self.needs_reduction:
                self.needs_reduction = False

                for name, param in self.module.named_parameters():
                    has_grad = param.grad is not None
                    # print('param name={} req grad={} has_grad={} size={}'.format(name, param.requires_grad, has_grad, param.size()))
                    if not param.requires_grad:
                        continue

                    # Count who has gradients for this parameter.
                    has_grad_count = torch.tensor(has_grad, dtype=torch.int64)  # [0. or 1.]
                    dist.all_reduce(has_grad_count, op=dist.ReduceOp.SUM) # [0. to world_size]

                    # Skip the reduction if no one has a gradient.
                    if has_grad_count == 0:
                        # print('has_grad_count==0')
                        continue

                    # print('has_grad_count={}'.format(has_grad_count))

                    # if has_grad:
                    #     first_el = param.grad.view(-1)[0]
                    #     print('first_el before reduction={}'.format(first_el))

                    if has_grad:
                        grad_data = param.grad.data
                    else:
                        # Create fake zeroed gradient data.
                        grad_data = torch.zeros_like(param)

                    dist.all_reduce(grad_data, op=dist.ReduceOp.SUM)
                    grad_data /= has_grad_count

                    # if has_grad:
                    #     first_el = param.grad.view(-1)[0]
                    #     print('first_el after reduction={}'.format(first_el))

        for param in list(self.module.parameters()):
            @torch.utils.hooks.unserializable_hook
            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(allreduce_params)

            if param.requires_grad:
                param.register_hook(allreduce_hook)

    def sync_parameters(self):
        logger.info('DistributedDataParallelSparseParamCPU::sync_parameters')
        for param in self.module.parameters():
            dist.broadcast(param.data, 0)

    def forward(self, *inputs, **kwargs):
        # print('DistributedDataParallelSparseParamCPU::forward')
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)
