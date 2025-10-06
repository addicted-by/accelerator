import torch
from typing import Tuple
import torch.nn.functional as F



def gaussian(window_size, sigma):
    ksize_half = (window_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=window_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples::

    #    >>> kornia.image.get_gaussian_kernel(3, 2.5)
    #    tensor([0.3243, 0.3513, 0.3243])

    #    >>> kornia.image.get_gaussian_kernel(5, 1.5)
    #    tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("kernel_size must be an odd positive integer. "
                        "Got {}".format(kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d



def get_gaussian_kernel2d(kernel_size: Tuple[int, int],
                          sigma: Tuple[float, float]) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

    #    >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
    #    tensor([[0.0947, 0.1183, 0.0947],
    #            [0.1183, 0.1478, 0.1183],
    #            [0.0947, 0.1183, 0.0947]])
    #
    #    >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
    #    tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
    #            [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
    #            [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError("kernel_size must be a tuple of length two. Got {}"
                        .format(kernel_size))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


def gaussian_blur(img, kernel_size, sigma):
    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")
    
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
        
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    
    kernel = get_gaussian_kernel2d(kernel_size, sigma).to(dtype=img.dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel, groups=img.shape[-3])
    
    return img