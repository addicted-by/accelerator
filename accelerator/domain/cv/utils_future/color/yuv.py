from typing import Tuple
import torch
from .matrices import get_torch_conversion_matrix
from .validation import (
    _validate_tensor_type, 
    _validate_channel_dim,
    _validate_shape_evenly_divisible2
)


def rgb_to_yuv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB image to YUV color space.
    
    Args:
        rgb: Input RGB image tensor with shape (..., 3, H, W)
            
    Returns:
        YUV image tensor with same shape as input
        
    Raises:
        ValueError: If input tensor type, shape or channel dimension is invalid
    """
    _validate_tensor_type(rgb)
    _validate_channel_dim(rgb, -3, 3)
    
    mat = get_torch_conversion_matrix('rgb', 'yuv', dtype=rgb.dtype, device=rgb.device)
    return (rgb.transpose(-3, -1) @ mat).transpose(-3, -1)


def yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
    """Convert YUV image to RGB color space.
    
    Args:
        yuv: Input YUV image tensor with shape (..., 3, H, W)
            
    Returns:
        RGB image tensor with same shape as input
        
    Raises:
        ValueError: If input tensor type, shape or channel dimension is invalid
    """
    _validate_tensor_type(yuv)
    _validate_channel_dim(yuv, -3, 3)

    mat = get_torch_conversion_matrix('yuv', 'rgb', dtype=yuv.dtype, device=yuv.device)
    return (yuv.transpose(-3, -1) @ mat).transpose(-3, -1)


def rgb_to_yuv420(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert RGB image to YUV420 format (Y at full resolution, U and V at quarter resolution).
    
    The input image dimensions must be divisible by 2.
    
    Args:
        rgb: Input RGB image tensor with shape (..., 3, H, W)
            
    Returns:
        Tuple containing:
            - Y channel tensor with shape (..., 1, H, W)
            - UV channels tensor with shape (..., 2, H/2, W/2)
        
    Raises:
        ValueError: If input tensor type, shape, channel dimension is invalid
                   or if image dimensions are not divisible by 2
    """
    _validate_tensor_type(rgb)
    _validate_channel_dim(rgb, -3, 3)
    _validate_shape_evenly_divisible2(rgb)

    yuv = rgb_to_yuv(rgb)
    y_channel = yuv[..., 0:1, :, :]
    
    uv_channels = yuv[..., 1:3, :, :]
    uv_downsampled = uv_channels.unfold(-2, 2, 2).unfold(-2, 2, 2).mean((-1, -2))
    
    return y_channel, uv_downsampled


def rgb_to_yuv422(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert RGB image to YUV422 format (Y at full resolution, U and V at half horizontal resolution).
    
    The input image width must be divisible by 2.
    
    Args:
        rgb: Input RGB image tensor with shape (..., 3, H, W)
            
    Returns:
        Tuple containing:
            - Y channel tensor with shape (..., 1, H, W)
            - UV channels tensor with shape (..., 2, H, W/2)
        
    Raises:
        ValueError: If input tensor type, shape, channel dimension is invalid
                   or if image width is not divisible by 2
    """
    _validate_tensor_type(rgb)
    _validate_channel_dim(rgb, -3, 3)
    _validate_shape_evenly_divisible2(rgb)

    yuv = rgb_to_yuv(rgb)
    y_channel = yuv[..., 0:1, :, :]
    
    uv_channels = yuv[..., 1:3, :, :]
    uv_downsampled = uv_channels.unfold(-1, 2, 2).mean(-1)
    
    return y_channel, uv_downsampled


def yuv420_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Convert YUV420 format to RGB image.
    
    Args:
        y: Y channel tensor with shape (..., 1, H, W)
        uv: UV channels tensor with shape (..., 2, H/2, W/2)
            
    Returns:
        RGB image tensor with shape (..., 3, H, W)
        
    Raises:
        ValueError: If input tensor types, shapes or channel dimensions are invalid
                   or if Y channel dimensions are not divisible by 2
    """
    _validate_tensor_type(y)
    _validate_tensor_type(uv)
    _validate_channel_dim(y, -3, 1)
    _validate_channel_dim(uv, -3, 2)
    _validate_shape_evenly_divisible2(y)
    
    # Check that dimensions are consistent
    if y.shape[-2] != uv.shape[-2] * 2 or y.shape[-1] != uv.shape[-1] * 2:
        raise ValueError(
            f"Y dimensions ({y.shape[-2]}, {y.shape[-1]}) must be twice the UV dimensions "
            f"({uv.shape[-2]}, {uv.shape[-1]}) for YUV420 format"
        )

    uv_upsampled = uv.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    yuv444 = torch.cat([y, uv_upsampled], dim=-3)
    
    return yuv_to_rgb(yuv444)


def yuv422_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Convert YUV422 format to RGB image.
    
    Args:
        y: Y channel tensor with shape (..., 1, H, W)
        uv: UV channels tensor with shape (..., 2, H, W/2)
            
    Returns:
        RGB image tensor with shape (..., 3, H, W)
        
    Raises:
        ValueError: If input tensor types, shapes or channel dimensions are invalid
                   or if Y channel width is not divisible by 2
    """
    _validate_tensor_type(y)
    _validate_tensor_type(uv)
    _validate_channel_dim(y, -3, 1)
    _validate_channel_dim(uv, -3, 2)
    _validate_shape_evenly_divisible2(y)
    
    if y.shape[-2] != uv.shape[-2] or y.shape[-1] != uv.shape[-1] * 2:
        raise ValueError(
            f"Y dimensions ({y.shape[-2]}, {y.shape[-1]}) must match UV dimensions "
            f"({uv.shape[-2]}, {uv.shape[-1] * 2}) for YUV422 format"
        )

    uv_upsampled = uv.repeat_interleave(2, dim=-1)
    yuv444 = torch.cat([y, uv_upsampled], dim=-3)
    
    return yuv_to_rgb(yuv444)