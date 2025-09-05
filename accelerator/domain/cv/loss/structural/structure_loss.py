import torch
from torch import nn
from functools import partial
import torch.nn.functional as F


from accelerator.domain.cv.utils import get_gaussian_kernel2d, smape_lp
from accelerator.runtime.loss.registry import registry, LossType


@registry.register_loss(LossType.IMG2IMG, name='structure_loss')
class StructureLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        window_size: int = 11,
        loss_fn: str = 'l1',
        smape_p: int = 1,
        smape_eps: float = 1e-7,
        
        **kwargs
    ):
        super().__init__()
        # super(StructureLoss, self).__init__()
        
        self.smape_p = smape_p
        self.smape_eps = smape_eps
        self.loss_fn = loss_fn
        
        loss_fns = {
            'l1': lambda x, y: (x - y).abs(),
            'l2': lambda x, y: (x - y).pow(2),
            'smape': partial(smape_lp, p=self.smape_p, eps=self.smape_eps)
        }
        if self.loss_fn in loss_fns:
            self.loss_fn = loss_fns[self.loss_fn]
        else:
            raise NotImplementedError(
                f"Loss function {self.loss_fn} does not implemented yet!"
            )
            
        self.structure = Structure(window_size, self.loss_fn)
    

    def forward(self, net_result, ground_truth, *args, **kwargs):
        return self.structure(net_result, ground_truth).mean()
        
    

class Structure(nn.Module):
    
    def __init__(
        self,
        window_size,
        loss_fn
    ) -> None:
        
        super(Structure, self).__init__()
        
        self.window_size = window_size
        self.loss_fn = loss_fn

        self.window = get_gaussian_kernel2d((window_size, window_size), (1.5, 1.5))
        self.padding = self.compute_zero_padding(window_size)

    @staticmethod
    def compute_zero_padding(kernel_size):
        return (kernel_size - 1) // 2

    def filter2D(self, input, kernel, channel):
        return F.conv2d(input, kernel, padding=self.padding, groups=channel)

    def forward(self, img1, img2):

        _, c, _, _ = img1.shape
        tmp_kernel= self.window.to(img1.device).to(img1.dtype)
        kernel = tmp_kernel.repeat(c, 1, 1, 1)

        mu1 = self.filter2D(img1, kernel, c)
        mu2 = self.filter2D(img2, kernel, c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.filter2D(img1 * img1, kernel, c) - mu1_sq
        sigma2_sq = self.filter2D(img2 * img2, kernel, c) - mu2_sq
        sigma12 = self.filter2D(img1 * img2, kernel, c) - mu1_mu2

        sigma12_sq_sign = torch.sign(sigma12) * sigma12.pow(2)
        sigma11_22_sq = sigma1_sq * sigma2_sq
        
        loss = self.loss_fn(sigma11_22_sq, sigma12_sq_sign)
        
        return loss


def structure_loss(img1, img2, window_size, loss_fn):
    return Structure(window_size, loss_fn)(img1, img2)