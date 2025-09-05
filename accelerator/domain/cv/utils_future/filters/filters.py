import numpy as np
import torch
from scipy.signal import convolve2d
import functools
import cv2


def _gaussian_kernel(size=5, sigma=1.0):
    """Generate a Gaussian kernel."""
    kernel = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    kernel = np.outer(kernel, kernel).astype(np.float32)
    kernel = kernel / np.sum(kernel)
    return kernel

def _laplace_kernel():
    """Generate a Laplace kernel."""
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    return kernel

def _sobel_kernel():
    """Generate a Sobel kernel for horizontal edge detection."""
    kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    return kernel

def _prewitt_kernel():
    """Generate a Prewitt kernel for horizontal edge detection."""
    kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    return kernel

def _unsharp_mask_kernel(size=5, sigma=1.0, amount=1.0):
    """Generate an Unsharp Mask kernel."""
    gaussian = _gaussian_kernel(size, sigma)
    unsharp_kernel = (1 + amount) * np.identity(size).reshape(size, size) - amount * gaussian
    return unsharp_kernel

_FILTER_GENERATORS = {
    'gaussian': _gaussian_kernel,
    'laplace': _laplace_kernel,
    'sobel': _sobel_kernel,
    'prewitt': _prewitt_kernel,
    'unsharp': _unsharp_mask_kernel,
}

@functools.lru_cache(maxsize=8)
def get_filter_kernel(filter_type, size=5, sigma=1.0, amount=1.0):
    """Get a kernel for a specified image processing filter.
    
    Kernels are generated only when requested and cached for future use.
    
    Args:
        filter_type: String representing the filter type (lowercase)
        size: Size of the kernel (for applicable filters)
        sigma: Standard deviation for Gaussian-based filters
        amount: Amount of sharpening for Unsharp Mask
        
    Returns:
        numpy.ndarray: The filter kernel
        
    Raises:
        ValueError: If the filter type is not available
    """
    filter_type = filter_type.lower()
    
    if filter_type not in _FILTER_GENERATORS:
        raise ValueError(f"Filter '{filter_type}' is not available")
    
    if filter_type == 'gaussian' or filter_type == 'unsharp':
        return _FILTER_GENERATORS[filter_type](size, sigma, amount)
    elif filter_type == 'laplace' or filter_type == 'sobel' or filter_type == 'prewitt':
        return _FILTER_GENERATORS[filter_type]()
    else:
        raise ValueError(f"Filter '{filter_type}' is not available")

@functools.lru_cache(maxsize=16)
def get_torch_filter_kernel(filter_type, size=5, sigma=1.0, amount=1.0, dtype=None, device=None):
    kernel = get_filter_kernel(filter_type, size, sigma, amount)
    return torch.tensor(kernel, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

def apply_filter(image, filter_type, size=5, sigma=1.0, amount=1.0, dtype=None, device=None):
    """Apply a specified filter to an image.
    
    Args:
        image: Input image as numpy.ndarray or torch.Tensor
        filter_type: String representing the filter type (lowercase)
        size: Size of the kernel (for applicable filters)
        sigma: Standard deviation for Gaussian-based filters
        amount: Amount of sharpening for Unsharp Mask
        dtype: Data type for PyTorch operations (optional)
        device: Device for PyTorch operations (optional)
        
    Returns:
        numpy.ndarray or torch.Tensor: Filtered image
        
    Raises:
        ValueError: If the filter type is not available
    """
    if isinstance(image, np.ndarray):
        kernel = get_filter_kernel(filter_type, size, sigma, amount)
        if len(image.shape) == 3:
            output = np.zeros_like(image)
            for i in range(3):
                output[:, :, i] = convolve2d(image[:, :, i], kernel, mode='same')
            return output
        else:
            return convolve2d(image, kernel, mode='same')
    elif isinstance(image, torch.Tensor):
        kernel = get_torch_filter_kernel(filter_type, size, sigma, amount, dtype, device)
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)
        output = torch.nn.functional.conv2d(image, kernel, padding=(kernel.shape[-2]//2, kernel.shape[-1]//2))
        output = output.squeeze(0)
        if output.shape[0] == 1:
            output = output.squeeze(0)
        return output
    else:
        raise ValueError("Unsupported image type. Must be numpy.ndarray or torch.Tensor.")