import torch
from accelerator.runtime.loss.registry import registry, LossType


@registry.register_loss(LossType.CUSTOM, 'rgb_std_loss')
def rgb_std_loss(net_result, ground_truth, *_, use_abs: bool=False, **kwargs):
    if use_abs:
        loss_value = torch.mean(torch.std((net_result - ground_truth).abs().mean(dim=(2,3)), dim=(1)))
    else:
        loss_value = torch.mean(torch.std((net_result - ground_truth).mean(dim=(2,3)), dim=(1)))
    
    return loss_value