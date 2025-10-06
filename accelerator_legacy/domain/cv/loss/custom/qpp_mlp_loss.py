import torch
from accelerator.utilities.model_utils import unwrap_model
from accelerator.runtime.loss.registry import registry, LossType
from accelerator.typings.base import CONTEXT


@registry.register_loss(LossType.CUSTOM, 'qpp_mlp_loss')
def qpp_mlp_loss(*_, context: CONTEXT, type_a=1, **kwargs):
    if context is None or context.model is None:
        raise ValueError(
            """
            qpp_mlp_loss requires the `context` with set `model` to be passed.
            """
        )
    if type_a == 1:
        loss_value = torch.abs(unwrap_model(context.model).qpp.mlp.a).mean()
    else:
        raise NotImplementedError()

    return loss_value