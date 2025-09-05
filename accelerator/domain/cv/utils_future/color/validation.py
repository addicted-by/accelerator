import torch
from textwrap import dedent


def _validate_tensor_type(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f'Input type is not a Tensor. {type(tensor)}')

def _validate_channel_dim(tensor, channel_dim, required_ch):
    if tensor.ndim < 3 or tensor.shape[channel_dim] != required_ch:
        raise ValueError(dedent(
            f'''
            Input data has incorrect shape: {tensor.shape}. 
            Expected (*, {required_ch}, h, w)')
            '''
        ))

def _validate_shape_evenly_divisible2(tensor):
    if tensor.ndim < 2 or tensor.shape[-2] % 2 == 1 or tensor.shape[-1] % 2 == 1:
        raise ValueError(f'Spatial dimensions (*,*,h,w) should be divisible by 2. Got: {tensor.shape}')