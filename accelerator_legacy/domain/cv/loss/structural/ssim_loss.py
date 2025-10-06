import torch
from .ssim import SSIM
from accelerator.runtime.loss import registry, LossType


@registry.register_loss(LossType.IMG2IMG, name='ssim_loss')
class SSIMLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        window_size: int = 11,
        reduction: str = 'mean',
        max_val: float = 1.,
        alpha: float = 1,
        beta: float = 1,
        gamma: float = 1,
        **kwargs
    ):
        super().__init__()
        # super(SSIMLoss, self).__init__()
        
        self.ssim = SSIM(
            window_size, 
            reduction,
            max_val,
            alpha=alpha, 
            beta=beta, 
            gamma=gamma
        )
        
    def forward(self, net_result, ground_truth, *args, **kwargs):
        return self.ssim(net_result, ground_truth)