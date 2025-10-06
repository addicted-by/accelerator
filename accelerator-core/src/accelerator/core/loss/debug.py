from omegaconf import DictConfig
import torch
from dataclasses import dataclass
from typing import Optional
import pathlib

from .errors import LossConfigurationError

from accelerator.utilities.default_config import _DefaultConfig 
from accelerator.utilities.logging import get_logger


logger = get_logger(__name__)


@dataclass
class DebugConfig(_DefaultConfig):
    """Configuration for debug features."""
    enabled: bool = False
    dump_path: Optional[str] = None
    save_frequency: int = 1
    max_files: int = 1000
    
    def __post_init__(self):
        if self.enabled and not self.dump_path:
            raise LossConfigurationError("dump_path must be provided when debug is enabled")



class DebugManager:
    """Manages debug functionality when enabled."""
    
    def __init__(self, config: DebugConfig):
        self.config = DictConfig(_DefaultConfig.create(config))
        self._file_count = 0
    
    def save_tensor(self, tensor: torch.Tensor, name: str, step: int = None) -> None:
        """Save tensor for debugging if enabled."""
        if not self.config.enabled:
            return
            
        if step is not None and step % self.config.save_frequency != 0:
            return
            
        if self._file_count >= self.config.max_files:
            logger.warning("Debug file limit reached, skipping save")
            return
        
        try:
            dump_path = pathlib.Path(self.config.dump_path)
            dump_path.mkdir(mode=0o777, parents=True, exist_ok=True)
            
            if isinstance(tensor, torch.Tensor):
                tensor_data = tensor.detach().cpu()
            else:
                tensor_data = tensor
                
            filename = f"{self._file_count:05d}_{name}.pt"
            filepath = dump_path / filename
            torch.save(tensor_data, filepath)
            
            self._file_count += 1
            
        except Exception as e:
            logger.error(f"Failed to save debug tensor: {e}")
