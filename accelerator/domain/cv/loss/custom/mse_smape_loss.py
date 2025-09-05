import torch
from accelerator.domain.cv.utils import smape_lp
from accelerator.runtime.loss import LossWrapper
from accelerator.utilities import get_logger


log = get_logger(__name__)


class MSESmapeLoss(LossWrapper):
    def calculate_batch_loss(
        self, 
        net_result, 
        ground_truth,
        transforms_info=None,
        *args, 
        **kwargs
    ):
        pca_asy = None
        
        if transforms_info:
            for (t_name, t_info) in transforms_info:
                if 'pca_asy' in t_info:
                    pca_asy = t_info['pca_asy']
                    break
        
        if pca_asy is None:
            log.warning('If you are not setting the `transform` returns `pca_asy` why are you using these implementation?')
            pca_asy = torch.ones_like(ground_truth, device=ground_truth.device)
        
        
        smape = smape_lp(ground_truth, net_result, p=self._cfg['p'], eps=self._cfg['eps'])
        return torch.mean(pca_asy * smape)   