from accelerator.runtime.loss import registry, LossType
from accelerator.domain.cv.utils import sobel, smape_lp


@registry.register_loss(LossType.IMG2IMG, name='sobel_smape_loss')
def sobel_smape_loss( 
    net_result, 
    ground_truth, 
    smape_p: int=1,
    *_, 
    **kwargs
):
    return smape_lp(
        sobel(net_result), 
        sobel(ground_truth),
        p=smape_p
    ).mean()