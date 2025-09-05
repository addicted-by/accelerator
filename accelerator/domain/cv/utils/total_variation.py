import torch
from torch import nn
import torch.nn.functional as F


class TVOperator(nn.Module):
    def __init__(self, weight=1.0):
        super(TVOperator, self).__init__()
        self.weight = weight

    def forward(self, x):
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)
        # h_x = x.size()[-2]
        # w_x = x.size()[-1]
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
        
        # print(
        #     f'x: {x.mean().item()}',
        #     f'right: {right.mean().item()}',
        #     f'bottom: {bottom.mean().item()}'
        # )
        dx, dy = right - left, bottom - top
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        dx_dy = torch.stack([dx, dy], dim=-1)
        return self.weight * dx_dy


def tv(input: torch.Tensor) -> torch.Tensor:
    r"""Computes the TV operator and returns the magnitude per channel.
    """
    return TVOperator()(input)