from __future__ import annotations

import torch
from textwrap import dedent
from accelerator.runtime.loss.registry import registry, LossType
from accelerator.typings.base import BatchTensorType, CONTEXT


@registry.register_loss(LossType.CUSTOM, 'wide_main_fusion_loss')
def wide_main_fusion_loss(
    net_result, 
    ground_truth, 
    *,
    inputs: BatchTensorType,
    context: CONTEXT, 
    p: int=1, 
    **kwargs
):
    model = context.model
    frames, meta, *_ = inputs

    if model is None or frames is None or meta is None:
        raise ValueError(dedent(
            f"""
            WideMainFusionLoss takes model, input and meta.
            Please, provide it:
            Model: {type(model)}
            input: {type(frames)}
            meta: {type(meta)}
            """
        ))
    
    frames[:, 25:49] = 0
    wo_main = model(frames, meta)
    
    if isinstance(wo_main, dict):
        wo_main = wo_main['prediction']
    if p == 1:
        loss_value = (wo_main - net_result).abs().mean()
    elif p == 2:
        loss_value = torch.pow((wo_main - net_result), 2).mean()
    else:
        raise NotImplementedError(f"{p} is not implemented!")
    
    return loss_value