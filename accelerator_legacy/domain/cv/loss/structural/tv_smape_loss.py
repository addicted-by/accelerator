import torch
from accelerator.domain.cv.utils import tv, smape_lp
from accelerator.runtime.loss import registry, LossType
from .ssim import SSIM


@registry.register_loss(LossType.IMG2IMG, name='tv_smape_loss')
class TVSmapeLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        window_size: int = 9,
        reduction: str = 'none',
        max_val: float = 1.,
        tv_smape_sq: bool = False,
        ssim_tv_smape: bool = False,
        smape_p: int = 1,
        **kwargs
    ):
        super().__init__()
        # super(TVSmapeLoss, self).__init__()
        
        self.tv_smape_sq = tv_smape_sq
        self.ssim_tv_smape = ssim_tv_smape
        self.smape_p = smape_p
        
        if self.ssim_tv_smape:
            self.ssim = SSIM(window_size, reduction=reduction, max_val=max_val) 
            
        
    def forward(self, net_result, ground_truth, *args, **kwargs):
        targets_tv = tv(ground_truth)
        model_outputs_tv = tv(net_result)
        
        if self.tv_smape_sq:
            targets_tv = targets_tv.pow(2)
            model_outputs_tv = model_outputs_tv.pow(2)
        
        if self.ssim_tv_smape:
            targets_tv = targets_tv.pow(2).sum(dim=-1)
            model_outputs_tv = model_outputs_tv.pow(2).sum(dim=-1)
            tv_smape = self.ssim(targets_tv, model_outputs_tv).mean()
        else:
            tv_smape = smape_lp(targets_tv, model_outputs_tv, p=self.smape_p).mean()
        
        return tv_smape