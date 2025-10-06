import numpy as np
import torch
import functools


def _rgb_to_xyz():
    return np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

def _rgb_to_yuv():
    return np.array([
        [0.299, -0.14714119, 0.61497538],
        [0.587, -0.28886916, -0.51496512],
        [0.114, 0.43601035, -0.10001026]
    ])

def _rgb_to_ycbcr():
    return np.array([
        [0.299, 0.7152, 0.0722],
        [-0.1146, -0.3854, 0.5],
        [0.5, -0.4542, -0.0458]
    ])

_MATRIX_GENERATORS = {
    ('rgb', 'xyz'): _rgb_to_xyz,
    ('rgb', 'yuv'): _rgb_to_yuv,
    ('rgb', 'ycbcr'): _rgb_to_ycbcr,
}

@functools.lru_cache(maxsize=8)
def get_conversion_matrix(source_space, target_space):
    """Get the conversion matrix between two color spaces.
    
    Matrices are generated only when requested and cached for future use.
    
    Args:
        source_space: String representing source color space (lowercase)
        target_space: String representing target color space (lowercase)
        
    Returns:
        numpy.ndarray: The conversion matrix
        
    Raises:
        ValueError: If the conversion matrix is not available
    """
    source = source_space.lower()
    target = target_space.lower()
    
    if (source, target) in _MATRIX_GENERATORS:
        return _MATRIX_GENERATORS[(source, target)]()
    
    if (target, source) in _MATRIX_GENERATORS:
        return np.linalg.inv(_MATRIX_GENERATORS[(target, source)]())
    
    raise ValueError(f"No conversion matrix available from {source_space} to {target_space}")

@functools.lru_cache(maxsize=8)
def get_torch_conversion_matrix(source_space, target_space, dtype=None, device=None):
    return torch.tensor(
        get_conversion_matrix(source_space, target_space),
        dtype=dtype,
        device=device
    )