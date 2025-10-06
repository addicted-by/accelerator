import torch
from torch import nn
import torch.nn.functional as F


def _get_lapl_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [-1., -1., -1.],
        [-1., 8., -1.],
        [-1., -1., -1.],
    ])

def lapl_all_dirs(input: torch.Tensor) -> torch.Tensor:
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    See :class:`~kornia.filters.SpatialGradient` for details.
    """
    return LaplAllDirections()(input)

class LaplAllDirections(nn.Module):
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    Return:
        torch.Tensor: the sobel edges of the input feature map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    Examples:
    #    >>> input = torch.rand(1, 3, 4, 4)
    #    >>> output = kornia.filters.SpatialGradient()(input)  # 1x3x2x4x4
    """
    def __init__(self) -> None:
        super(LaplAllDirections, self).__init__()
        self.kernel: torch.Tensor = self.get_lapl_kernel()

    @staticmethod
    def get_lapl_kernel() -> torch.Tensor:
        kernel: torch.Tensor = _get_lapl_kernel_3x3()
        return kernel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1, 1)

        # convolve input tensor with sobel kernel
        kernel_flip: torch.Tensor = kernel.flip(-3)
        return F.conv3d(input[:, :, None], kernel_flip, padding=1, groups=c)


def laplace(input: torch.Tensor) -> torch.Tensor:
    r"""Computes the Sobel operator and returns the magnitude per channel.

    See :class:`~kornia.filters.Sobel` for details.
    """
    return Laplace()(input)



class Laplace(nn.Module):
    r"""Computes the Sobel operator and returns the magnitude per channel.

    Return:
        torch.Tensor: the sobel edge gradient maginitudes map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
    #    >>> input = torch.rand(1, 3, 4, 4)
    #    >>> output = kornia.filters.Sobel()(input)  # 1x3x4x4
    """
    def __init__(self) -> None:
        super(Laplace, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # compute the x/y gradients
        edges: torch.Tensor = lapl_all_dirs(input)

        # unpack the edges
        # gx: torch.Tensor = edges[:, :, 0]
        # gy: torch.Tensor = edges[:, :, 1]

        # compute gradient maginitude
        # magnitude: torch.Tensor = torch.stack([gx, gy], dim=-1) / 4 # torch.sqrt(gx * gx + gy * gy)
        magnitude: torch.Tensor = edges / 8 # torch.sqrt(gx * gx + gy * gy)
        return magnitude
