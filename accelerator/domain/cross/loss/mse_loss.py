from accelerator.runtime.loss import LossType, registry


@registry.register_loss(LossType.REGRESSION, 'mse_loss_fn')
def mse_loss(net_result, ground_truth, *args, **kwargs):
    return (net_result - ground_truth).pow(2).mean()


@registry.register_loss(LossType.REGRESSION, 'multi_mse_loss_fn')
def multi_mse_loss_fn(net_result, ground_truth, *args, **kwargs):
    loss = 0.
    for nr, gt in zip(net_result, ground_truth):
        loss += (nr - gt).pow(2).mean()
    return loss