"""
This script can be used for weight surgery; in particular, weight expansion.

The biases will be initialized to zero, and the weights will be initialized as kaiming uniform.
Only linear layers are implemented at the moment. The target shapes need to be strictly bigger
than their original.
"""

from collections import namedtuple
import math
import torch

Surgery = namedtuple('Surgery', ['name', 'shape'])

# Prepare the operating theatre.
pretrained_model = 'exp15-job1-model_000001640.pt'
output_name = 'patient.pt'
procedures = [
    Surgery(name='affine_unit_basic_stats.weight', shape=[128, 12]),
    Surgery(name='affine_head_enum.weight', shape=[4, 256]),
    Surgery(name='affine_head_enum.bias', shape=[4]),
]

# Bring in the patient.
m = torch.load(pretrained_model)

# Go through the procedures.
for p in procedures:
    name = p.name
    assert name in m
    t = m[name]
    source_shape = t.shape
    target_shape = torch.Size(p.shape)
    shape_diff = torch.tensor(target_shape) - torch.tensor(source_shape)
    if shape_diff.sum() == 0:
        raise ValueError('Requested shape ({}) of {} is identical.'.format(target_shape, name))
    if (shape_diff < 0).any():
        raise ValueError('At least one requested shape ({}) is less than the source ({}).'.format(
            target_shape, source_shape))

    # Create a new bigger organ, to transplant the old one into.
    x = torch.nn.Parameter(data=torch.zeros(target_shape))

    if '.weight' in name:
        torch.nn.init.kaiming_uniform_(x, a=math.sqrt(5))
    elif '.bias' in name:
        pass
    else:
        ValueError("Need either '.weight' or '.bias' in the name ({})".format(name))

    # And transplant back the original organs in the new, bigger one.
    if len(target_shape) == 1:
        x[:source_shape[0]] = t
    else:
        x[:source_shape[0], :source_shape[1]] = t
    x = x.detach()

    # And put it back into the patient.
    m[name] = x

# Release the patient.
torch.save(m, output_name)
