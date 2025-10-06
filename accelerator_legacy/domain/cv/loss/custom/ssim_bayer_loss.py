from ..structural.ssim import ssim

from accelerator.typings.base import _DEVICE
from accelerator.domain.cv.utils import rgb_to_bayer
from accelerator.runtime.loss.registry import registry, LossType


@registry.register_loss(LossType.CUSTOM, 'ssim_bayer_loss')
def ssim_bayer_loss(
    net_result, 
    ground_truth, 
    *_, 
    device: _DEVICE, 
    per_pixel: bool = False, 
    window_size: int = 11,
    **kwargs
):
    bayer_net_result = rgb_to_bayer(
        net_result, 
        per_pixel=per_pixel
    )
    bayer_gt = rgb_to_bayer(
        ground_truth, 
        per_pixel=per_pixel
    )

    ssim_bayer_loss = ssim(
        bayer_net_result, 
        bayer_gt, 
        window_size=window_size, 
        reduction='none', 
        max_val=1.0
    ).mean()
    return ssim_bayer_loss