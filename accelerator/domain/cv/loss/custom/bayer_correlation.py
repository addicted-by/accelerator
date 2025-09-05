import torch
from ..structural.structure_loss import structure_loss

from accelerator.runtime.loss.registry import registry, LossType
from accelerator.domain.cv.utils import rgb_to_bayer


def correlationPearson(img1, img2, window_size):
    def correlation(sigma11_22_sq, sigma12_sq_sign):
        return (
            torch.clamp(
                torch.tensor(1.0) - sigma12_sq_sign / (sigma11_22_sq + 1e-16),
                min=0,
                max=2,
            )
            / 2.0
        )

    return structure_loss(img1, img2, window_size, correlation)


@registry.register_loss(LossType.CUSTOM, "bayer_correlation_loss")
def bayer_correlation_loss(
    net_result, ground_truth, per_pixel=False, window_size=11, *args, **kwargs
):
    bayer_net_result = rgb_to_bayer(net_result, per_pixel=per_pixel)
    bayer_gt = rgb_to_bayer(ground_truth, per_pixel=per_pixel)

    bayer_correlation_loss = correlationPearson(
        bayer_net_result, bayer_gt, window_size=window_size
    ).mean()
    return bayer_correlation_loss
