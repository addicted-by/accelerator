import torch
from accelerator.domain.cv.utils import tv, smooth_loss_fn
from accelerator.runtime.loss import registry, LossType


@registry.register_loss(LossType.IMG2IMG, name='tv_loss')
class TVLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        smooth_loss_type: str='l1',
        beta: float=1,
        **kwargs
    ):
        super().__init__()
        
        self.smooth_loss = smooth_loss_fn(smooth_loss_type, beta)
        
    def forward(self, net_result, ground_truth, *args, **kwargs):
        return self.smooth_loss(tv(net_result), tv(ground_truth))