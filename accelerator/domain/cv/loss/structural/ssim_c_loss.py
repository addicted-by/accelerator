import torch
from .ssim import SSIM
from accelerator.runtime.loss import registry, LossType


@registry.register_loss(LossType.IMG2IMG, name='ssim_c_loss')
class SSIMCLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        window_size: int = 11,
        reduction: str = 'mean',
        max_val: float = 1.,
        **kwargs
    ):
        super().__init__()
        # super(SSIMCLoss, self).__init__()
        
        self.ssim = SSIM(
            window_size, 
            reduction,
            max_val,
            alpha=0, beta=0, gamma=1
        )
        
    def forward(self, net_result, ground_truth, *args, **kwargs):
        return self.ssim(net_result, ground_truth)