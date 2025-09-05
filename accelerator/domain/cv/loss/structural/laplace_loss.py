import torch
from accelerator.domain.cv.utils import laplace, smooth_loss_fn
from accelerator.runtime.loss import registry, LossType


@registry.register_loss(LossType.IMG2IMG, name='laplace_loss')
class LaplaceLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        smooth_loss_type: str='l1',
        beta: float=1,
        **kwargs
    ):
        super().__init__()
        # super(LaplaceLoss, self).__init__()
        
        self.smooth_loss = smooth_loss_fn(smooth_loss_type, beta)
        
    def forward(self, net_result, ground_truth, *args, **kwargs):
        return self.smooth_loss(laplace(net_result), laplace(ground_truth))