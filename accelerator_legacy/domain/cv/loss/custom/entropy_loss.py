import torch
from accelerator.runtime.loss.registry import registry, LossType


@registry.register_loss(LossType.CUSTOM, 'entropy_loss')
def entropy_loss(net_result, ground_truth, *args, **kwargs):
    return torch.log(
        (ground_truth - net_result)
        .std(dim=(1, 2, 3))
        .mean() + 1e-9
    )