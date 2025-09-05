import torch
from .ssim import ssim

from accelerator.runtime.loss import registry, LossType
from accelerator.domain.cv.utils import sobel, tv, calc_asym_gradient_mask


@registry.register_loss(LossType.IMG2IMG, name='psf_loss')
class PSFLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        asym_gradient_mask: bool = True,
        tv_coeff: float = 0.333333,
        sobel_coeff: float = 0.333333,
        ssim_coeff: float = 0.333333,
        **kwargs
    ):
        super().__init__()
        # super(PSFLoss, self).__init__()
        
        self.asym_gradient_mask = asym_gradient_mask
        self.tv_coeff = tv_coeff
        self.sobel_coeff = sobel_coeff
        self.ssim_coeff = ssim_coeff
        
    def forward(
        self,
        net_result,
        ground_truth,
        *args, **kwargs
    ):
        if self.asym_gradient_mask:
            target_tv_mask = tv(ground_truth)
            netout_tv_mask = tv(net_result)
            asym_gradient_mask = calc_asym_gradient_mask(netout_tv_mask, target_tv_mask)
            tv_diff_mask = torch.abs(target_tv_mask - netout_tv_mask) * asym_gradient_mask
            tv_loss = torch.mean(tv_diff_mask)
            sobel_loss = torch.mean(torch.abs(sobel(ground_truth) - sobel(net_result)))
            ssim_loss = torch.mean(ssim(ground_truth, net_result, window_size=9, reduction='none', max_val=1.0))
        else:
            tv_loss = torch.mean(torch.abs(tv(ground_truth) - tv(net_result)))
            sobel_loss = torch.mean(torch.abs(sobel(ground_truth) - sobel(net_result)))
            ssim_loss = torch.mean(ssim(ground_truth, net_result, window_size=9, reduction='none', max_val=1.0))
            
            
        psf_loss = (
            self.tv_coeff * tv_loss + 
            self.sobel_coeff * sobel_loss + 
            self.ssim_coeff * ssim_loss
        )
        return psf_loss