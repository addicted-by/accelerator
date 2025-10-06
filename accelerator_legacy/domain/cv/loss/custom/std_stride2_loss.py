import torch
from accelerator.domain.cv.utils import rggb_to_bayer
from accelerator.runtime.loss.registry import registry, LossType
from accelerator.typings.base import BatchTensorType


def calculate_std_stride2(net_result, frame):
    bayer = rggb_to_bayer(frame)

    output_R = bayer - net_result[:, 0, :, :].unsqueeze(1)
    output_G = bayer - net_result[:, 1, :, :].unsqueeze(1)
    output_B = bayer - net_result[:, 2, :, :].unsqueeze(1)

    return torch.mean(torch.std(torch.stack([
        output_R[:, :, 0::2, 0::2].mean(dim=(1,2,3)),
        output_G[:, :, 0::2, 1::2].mean(dim=(1,2,3)),
        output_B[:, :, 1::2, 1::2].mean(dim=(1,2,3))]
    ), dim=(0)))


@registry.register_loss(LossType.CUSTOM, 'std_stride2_loss')
def std_stride2_loss(
    net_result, 
    ground_truth, 
    *_,
    inputs: BatchTensorType,
    p: int = 1, 
    **kwargs
):
    frames, *_ = inputs
    if frames is None:
        raise ValueError(
            """
            std_stride2_loss requires frames to calculate
            the value! Please, provide it!
            """
        )
        
    net_std = calculate_std_stride2(net_result, frames[:, :4, ...]) # N0
    gt_std = calculate_std_stride2(ground_truth, frames[:, :4, ...])
    if p == 2:
        std_stride2_loss = torch.sqrt((net_std - gt_std)**2 + 1e-16)
    elif p == 1:
        std_stride2_loss = torch.abs(net_std - gt_std)
    else:
        raise ValueError(f"Not implemented p: {p}")
    

    return std_stride2_loss