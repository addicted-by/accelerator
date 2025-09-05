from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from accelerator.typings.base import ConfigType
from accelerator.utilities.default_config import _DefaultConfig, dataclass
from accelerator.utilities.logging import get_logger

from .base import DistributedBackend

logger = get_logger(__name__)


try:
    import accelerate
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    logger.warning("HuggingFace Accelerate not available. Install with: `pip install accelerate`")


@dataclass
class AccelerateEngineDefaults(_DefaultConfig):
    mixed_precision: str = 'no'
    gradient_accumulation_steps: int = 1

    device_placement: bool = True
    dynamo_backed: str = 'no'
    dataloader_config: Optional[ConfigType] = None
    deepspeed_plugin: Optional[ConfigType] = None
    fsdp_plugin: Optional[ConfigType] = None



class AccelerateEngine(DistributedBackend):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, default_config=AccelerateEngineDefaults)
        
        if not ACCELERATE_AVAILABLE:
            raise ImportError("HuggingFace Accelerate is required but not installed")
        
        self.accelerator = None

    def setup(self) -> None:
        accelerator_kwargs = dict(self.config)
        self.accelerator = Accelerator(**accelerator_kwargs)
        logger.info(f"Initialized Accelerate: device={self.accelerator.device}, "
                   f"num_processes={self.accelerator.num_processes}")
    
    @property
    def device(self):
        return self.accelerator.device
    
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.accelerator is None:
            raise RuntimeError("Must call setup() before preparing model")
        return self.accelerator.prepare(model)
    
    def prepare_dataloader(self, loader: DataLoader) -> DataLoader:
        if self.accelerator is None:
            raise RuntimeError("Must call setup() before preparing dataloader")
        return self.accelerator.prepare(loader)
    
    def prepare_optimizer(self, optimizer) -> Any:
        if self.accelerator is None:
            raise RuntimeError("Must call setup() before preparing optimizer")
        return self.accelerator.prepare(optimizer)
    
    def rank(self) -> int:
        if self.accelerator is None:
            return 0
        return self.accelerator.process_index
    
    def world_size(self) -> int:
        if self.accelerator is None:
            return 1
        return self.accelerator.num_processes
    
    def is_main_process(self) -> bool:
        if self.accelerator is None:
            return True
        return self.accelerator.is_main_process
    
    def barrier(self) -> None:
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
    
    def all_reduce(self, tensor, op='mean') -> torch.Tensor:
        if self.accelerator is None:
            return tensor
        
        if op == 'mean':
            return self.accelerator.reduce(tensor, reduction='mean')
        elif op == 'sum':
            return self.accelerator.reduce(tensor, reduction='sum')
        else:
            raise ValueError(f"Unsupported reduction operation: {op}")
    
    def gather(self, tensor) -> Any:
        if self.accelerator is None:
            return [tensor]
        return self.accelerator.gather(tensor)
    
    def spawn(self, fn: Callable, *args) -> None:
        fn(*args)
    
    def cleanup(self) -> None:
        if self.accelerator is not None:
            self.accelerator.end_training()
            logger.info("Cleaned up Accelerate")
    
    def backward(self, loss: torch.Tensor) -> None:
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
    
    def clip_grad_norm(self, parameters, max_norm: float) -> None:
        if self.accelerator is not None:
            self.accelerator.clip_grad_norm_(parameters, max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    
    def save_state(self, output_dir: str) -> None:
        if self.accelerator is not None:
            self.accelerator.save_state(output_dir)
    
    def load_state(self, input_dir: str) -> None:
        if self.accelerator is not None:
            self.accelerator.load_state(input_dir)

    def broadcast(self, obj, src = 0):
        return accelerate.utils.broadcast(obj, src)