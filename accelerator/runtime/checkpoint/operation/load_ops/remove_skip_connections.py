from accelerator.runtime.checkpoint.operation.registry import registry, OperationType
from accelerator.utilities.logging import get_logger
import torch


log = get_logger(__name__)


@registry.register_operation([OperationType.PRE_LOAD_OPS, OperationType.POST_LOAD_OPS])
def remove_skip_connections(model: torch.nn.Module, **kwargs):
    log.info("Removing skip connections...")
    
    for name, module in model.named_modules():
        if hasattr(module, 'to_rep'):
            if module.is_rep:
                log.info(f'\t\t{name:50} - already removed')
            else:
                log.info(f'\t\t{name:50} - removing skip')
                module.to_rep()

            module.is_rep_cpu = module.is_rep.detach().item()