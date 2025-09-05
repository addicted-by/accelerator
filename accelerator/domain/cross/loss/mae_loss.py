import torch
from accelerator.runtime.loss import LossType, registry


@registry.register_loss(LossType.REGRESSION, 'mae_loss_fn')
def mae_loss(net_result, ground_truth, *args, **kwargs):
    return torch.mean(torch.abs(net_result - ground_truth))