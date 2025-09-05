import torch

from accelerator.runtime.loss import registry, LossType
from accelerator.domain.cv.utils import sobel, smooth_loss_fn


@registry.register_loss(LossType.IMG2IMG, name='sobel_loss')
class SobelLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        smooth_loss_type: str='l1',
        beta: float=1,
        **kwargs
    ):
        super().__init__()
        # super(SobelLoss, self).__init__()
        
        self.smooth_loss = smooth_loss_fn(smooth_loss_type, beta)
        
    def forward(self, net_result, ground_truth, *args, **kwargs):
        return self.smooth_loss(sobel(net_result), sobel(ground_truth))