
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import tensorboardX
import jax

from policy import Policy

p = Policy()

dummy_input = (torch.Tensor(2), torch.Tensor(1,8),  torch.Tensor(1,8),  torch.Tensor(1,8), torch.Tensor(1,8))

outs, hidden = p(*dummy_input, hidden=None)

ins = dummy_input + (hidden,)
print(ins)

with SummaryWriter(comment='LinearInLinear') as w:
    w.add_graph(p, ins, verbose=True)
