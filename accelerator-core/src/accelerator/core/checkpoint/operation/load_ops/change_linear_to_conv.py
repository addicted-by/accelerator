import torch

from accelerator.runtime.checkpoint.operation.registry import OperationType, registry
from accelerator.utilities.logging import get_logger
from accelerator.utilities.model_utils import get_parent_module

log = get_logger(__name__)


def _conv1x1_from_linear(module: torch.nn.Linear):
    _bias = False if module.bias is None else True
    convolution = torch.nn.Conv2d(
        in_channels=module.in_features, out_channels=module.out_features, kernel_size=1, stride=1, padding=0, bias=_bias
    )
    convolution.weight.data = module.weight.data
    if _bias:
        convolution.bias.data = module.bias.data

    return convolution


@registry.register_operation([OperationType.PRE_LOAD_OPS, OperationType.POST_LOAD_OPS])
def change_linear_to_conv1x1(model: torch.nn.Module, **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parent_module, attribute = get_parent_module(model, name)
            convolution = _conv1x1_from_linear(module)
            log.info(f"Changing linear layer {name} to conv1x1")
            setattr(parent_module, attribute, convolution)


@registry.register_operation([OperationType.PRE_LOAD_OPS, OperationType.POST_LOAD_OPS])
def change_linear_specified_to_conv1x1(model: torch.nn.Module, modules: list[str], **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in modules:
            parent_module, attribute = get_parent_module(model, name)
            convolution = _conv1x1_from_linear(module)
            log.info(f"Changing linear layer {name} to conv1x1")
            setattr(parent_module, attribute, convolution)
