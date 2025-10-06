import torch
from typing import Any, Optional, Callable
from torch.utils.data import DataLoader

from .base import DistributedBackend
from accelerator.utilities.logging import get_logger
from accelerator.utilities.default_config import _DefaultConfig, dataclass
from accelerator.utilities.typings import ConfigType


logger = get_logger(__name__)


@dataclass
class SingleGPUEngineDefaults(_DefaultConfig):
    device: str = 'cuda' if torch.cuda.is_available else 'cpu'


class SingleGPUEngine(DistributedBackend):
    def __init__(self, config: Optional[ConfigType]):
        super().__init__(config, default_config=SingleGPUEngineDefaults)
        self._device = config.get('device')
        
    @property
    def device(self):
        return self._device
        
    def setup(self) -> None:
        if self._device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self._device = 'cpu'
        
        if self._device == 'cuda':
            torch.cuda.set_device(0)
            
        logger.info(f"Initialized SingleGPU engine with device: {self._device}")
    
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.to(self._device)
    
    def prepare_dataloader(self, loader: DataLoader) -> DataLoader:
        return loader
    
    def prepare_optimizer(self, optimizer) -> Any:
        return optimizer
    
    def rank(self) -> int:
        return 0
    
    def world_size(self) -> int:
        return 1
    
    def is_main_process(self) -> bool:
        return True
    
    def barrier(self) -> None:
        pass
    
    def all_reduce(self, tensor, op='mean') -> torch.Tensor:
        return tensor
    
    def gather(self, tensor) -> Any:
        return [tensor]
    
    def spawn(self, fn: Callable, *args) -> None:
        fn(*args)
    
    def cleanup(self) -> None:
        logger.info("Cleaned up SingleGPU engine")
    
    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()