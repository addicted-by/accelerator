from accelerator.runtime.loss import registry, LossType
from accelerator.domain.cv.utils import laplace, smape_lp


@registry.register_loss(LossType.IMG2IMG, name='laplace_smape_loss')
def laplace_smape_loss( 
    net_result, 
    ground_truth, 
    smape_p: int=1,
    *_, 
    **kwargs
):
    return smape_lp(
        laplace(net_result), 
        laplace(ground_truth),
        p=smape_p
    ).mean()