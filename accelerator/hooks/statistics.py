"""
Activation statistics collection hooks for neural network analysis.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class StatisticsConfig:
    """Configuration for activation statistics collection."""
    # Basic statistics
    collect_mean: bool = True
    collect_std: bool = True
    collect_min_max: bool = True
    collect_percentiles: List[float] = field(default_factory=lambda: [95, 99])
    
    # Advanced statistics
    collect_histogram: bool = False
    histogram_bins: int = 100
    collect_sparsity: bool = False
    
    # Performance settings
    update_frequency: int = 1  # Update every N forward passes
    max_samples: Optional[int] = 1000  # Limit samples for memory
    
    # Memory management
    max_memory_mb: Optional[float] = None


@dataclass
class TensorStatistics:
    """Statistics for a single tensor (input or output)."""
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    min: Optional[torch.Tensor] = None
    max: Optional[torch.Tensor] = None
    percentiles: Dict[float, torch.Tensor] = field(default_factory=dict)
    histogram: Optional[torch.Tensor] = None
    sparsity_ratio: Optional[float] = None
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None
    sample_count: int = 0


@dataclass
class ModuleStatistics:
    """Complete statistics for a module's input and output."""
    module_name: str
    input_stats: TensorStatistics = field(default_factory=TensorStatistics)
    output_stats: TensorStatistics = field(default_factory=TensorStatistics)
    total_samples: int = 0
    last_updated: Optional[datetime] = None


class OnlineStatistics:
    """Efficient online computation of mean and variance using Welford's algorithm."""
    
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None  # Sum of squares of differences from mean
        self.min_val = None
        self.max_val = None
    
    def update(self, tensor: torch.Tensor) -> None:
        """Update statistics with a new tensor."""
        # Flatten tensor for statistics computation
        flat_tensor = tensor.detach().flatten()
        
        if self.count == 0:
            self.mean = torch.zeros_like(flat_tensor[0])
            self.m2 = torch.zeros_like(flat_tensor[0])
            self.min_val = flat_tensor.min()
            self.max_val = flat_tensor.max()
        
        for value in flat_tensor:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2
            
            # Update min/max
            if value < self.min_val:
                self.min_val = value
            if value > self.max_val:
                self.max_val = value
    
    def get_mean(self) -> torch.Tensor:
        """Get current mean."""
        return self.mean.clone() if self.mean is not None else torch.tensor(0.0)
    
    def get_variance(self) -> torch.Tensor:
        """Get current variance."""
        if self.count < 2:
            return torch.tensor(0.0)
        return self.m2 / (self.count - 1)
    
    def get_std(self) -> torch.Tensor:
        """Get current standard deviation."""
        return torch.sqrt(self.get_variance())
    
    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current min and max values."""
        min_val = self.min_val if self.min_val is not None else torch.tensor(0.0)
        max_val = self.max_val if self.max_val is not None else torch.tensor(0.0)
        return min_val.clone(), max_val.clone()


class StatisticsCollector:
    """Collects and manages activation statistics for multiple modules."""
    
    def __init__(self, config: StatisticsConfig):
        self.config = config
        self.module_stats: Dict[str, ModuleStatistics] = {}
        self.online_stats: Dict[str, Dict[str, OnlineStatistics]] = {}  # module -> {input/output -> stats}
        self.sample_buffers: Dict[str, Dict[str, List[torch.Tensor]]] = {}  # For percentiles
    
    def update_statistics(self, module_name: str, input_tensor: torch.Tensor, 
                         output_tensor: torch.Tensor) -> None:
        """Update statistics for a module's input and output tensors."""
        if module_name not in self.module_stats:
            self.module_stats[module_name] = ModuleStatistics(module_name=module_name)
            self.online_stats[module_name] = {
                'input': OnlineStatistics(),
                'output': OnlineStatistics()
            }
            self.sample_buffers[module_name] = {'input': [], 'output': []}
        
        # Update online statistics
        self.online_stats[module_name]['input'].update(input_tensor)
        self.online_stats[module_name]['output'].update(output_tensor)
        
        # Store samples for percentiles (with memory limit)
        if self.config.collect_percentiles:
            self._store_samples(module_name, 'input', input_tensor)
            self._store_samples(module_name, 'output', output_tensor)
        
        # Update module statistics
        self._update_module_stats(module_name, input_tensor, output_tensor)
        
        self.module_stats[module_name].total_samples += 1
        self.module_stats[module_name].last_updated = datetime.now()
    
    def _store_samples(self, module_name: str, tensor_type: str, tensor: torch.Tensor) -> None:
        """Store tensor samples for percentile computation."""
        if self.config.max_samples is None:
            return
        
        buffer = self.sample_buffers[module_name][tensor_type]
        flat_tensor = tensor.detach().flatten()
        
        # Simple reservoir sampling
        if len(buffer) < self.config.max_samples:
            buffer.extend(flat_tensor.tolist())
        else:
            # Replace random samples
            import random
            for _ in range(min(len(flat_tensor), 100)):  # Limit updates per call
                idx = random.randint(0, len(buffer) - 1)
                buffer[idx] = random.choice(flat_tensor).item()
    
    def _update_module_stats(self, module_name: str, input_tensor: torch.Tensor, 
                           output_tensor: torch.Tensor) -> None:
        """Update the stored module statistics."""
        stats = self.module_stats[module_name]
        online_input = self.online_stats[module_name]['input']
        online_output = self.online_stats[module_name]['output']
        
        # Update input statistics
        if self.config.collect_mean:
            stats.input_stats.mean = online_input.get_mean()
        if self.config.collect_std:
            stats.input_stats.std = online_input.get_std()
        if self.config.collect_min_max:
            stats.input_stats.min, stats.input_stats.max = online_input.get_min_max()
        
        # Update output statistics
        if self.config.collect_mean:
            stats.output_stats.mean = online_output.get_mean()
        if self.config.collect_std:
            stats.output_stats.std = online_output.get_std()
        if self.config.collect_min_max:
            stats.output_stats.min, stats.output_stats.max = online_output.get_min_max()
        
        # Update percentiles
        if self.config.collect_percentiles:
            stats.input_stats.percentiles = self._compute_percentiles(module_name, 'input')
            stats.output_stats.percentiles = self._compute_percentiles(module_name, 'output')
        
        # Update sparsity
        if self.config.collect_sparsity:
            stats.input_stats.sparsity_ratio = self._compute_sparsity(input_tensor)
            stats.output_stats.sparsity_ratio = self._compute_sparsity(output_tensor)
        
        # Update metadata
        stats.input_stats.shape = tuple(input_tensor.shape)
        stats.input_stats.dtype = input_tensor.dtype
        stats.input_stats.sample_count = online_input.count
        
        stats.output_stats.shape = tuple(output_tensor.shape)
        stats.output_stats.dtype = output_tensor.dtype
        stats.output_stats.sample_count = online_output.count
    
    def _compute_percentiles(self, module_name: str, tensor_type: str) -> Dict[float, torch.Tensor]:
        """Compute percentiles from stored samples."""
        buffer = self.sample_buffers[module_name][tensor_type]
        if not buffer:
            return {}
        
        percentiles = {}
        buffer_tensor = torch.tensor(buffer)
        
        for p in self.config.collect_percentiles:
            percentiles[p] = torch.quantile(buffer_tensor, p / 100.0)
        
        return percentiles
    
    def _compute_sparsity(self, tensor: torch.Tensor) -> float:
        """Compute sparsity ratio (fraction of zero elements)."""
        total_elements = tensor.numel()
        zero_elements = (tensor == 0).sum().item()
        return zero_elements / total_elements if total_elements > 0 else 0.0
    
    def get_module_statistics(self, module_name: str) -> Optional[ModuleStatistics]:
        """Get statistics for a specific module."""
        return self.module_stats.get(module_name)
    
    def get_all_statistics(self) -> Dict[str, ModuleStatistics]:
        """Get statistics for all modules."""
        return self.module_stats.copy()
    
    def clear_statistics(self, module_name: Optional[str] = None) -> None:
        """Clear statistics for a specific module or all modules."""
        if module_name is None:
            self.module_stats.clear()
            self.online_stats.clear()
            self.sample_buffers.clear()
        else:
            self.module_stats.pop(module_name, None)
            self.online_stats.pop(module_name, None)
            self.sample_buffers.pop(module_name, None)
    
    def export_statistics(self, format: str = "json") -> Union[Dict, str]:
        """Export statistics in the specified format."""
        if format == "json":
            return self._export_json()
        elif format == "quantization":
            return self._export_quantization_data()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self) -> Dict[str, Any]:
        """Export statistics as JSON-serializable dictionary."""
        result = {}
        for module_name, stats in self.module_stats.items():
            result[module_name] = {
                'total_samples': stats.total_samples,
                'last_updated': stats.last_updated.isoformat() if stats.last_updated else None,
                'input_stats': self._tensor_stats_to_dict(stats.input_stats),
                'output_stats': self._tensor_stats_to_dict(stats.output_stats)
            }
        return result
    
    def _tensor_stats_to_dict(self, tensor_stats: TensorStatistics) -> Dict[str, Any]:
        """Convert TensorStatistics to dictionary."""
        result = {
            'sample_count': tensor_stats.sample_count,
            'shape': tensor_stats.shape,
            'dtype': str(tensor_stats.dtype) if tensor_stats.dtype else None,
            'sparsity_ratio': tensor_stats.sparsity_ratio
        }
        
        # Convert tensors to lists for JSON serialization
        if tensor_stats.mean is not None:
            result['mean'] = tensor_stats.mean.item()
        if tensor_stats.std is not None:
            result['std'] = tensor_stats.std.item()
        if tensor_stats.min is not None:
            result['min'] = tensor_stats.min.item()
        if tensor_stats.max is not None:
            result['max'] = tensor_stats.max.item()
        
        if tensor_stats.percentiles:
            result['percentiles'] = {
                str(p): v.item() for p, v in tensor_stats.percentiles.items()
            }
        
        return result
    
    def _export_quantization_data(self) -> Dict[str, Any]:
        """Export data in format suitable for quantization calibration."""
        result = {}
        for module_name, stats in self.module_stats.items():
            if stats.output_stats.min is not None and stats.output_stats.max is not None:
                result[module_name] = {
                    'min': stats.output_stats.min.item(),
                    'max': stats.output_stats.max.item(),
                    'percentiles': {
                        str(p): v.item() for p, v in stats.output_stats.percentiles.items()
                    } if stats.output_stats.percentiles else {}
                }
        return result


class ActivationStatisticsHook:
    """Hook function for collecting activation statistics."""
    
    def __init__(self, config: StatisticsConfig, collector: Optional[StatisticsCollector] = None):
        self.config = config
        self.collector = collector or StatisticsCollector(config)
        self.call_count = 0
    
    def __call__(self, module: nn.Module, input: Tuple[torch.Tensor, ...], 
                 output: torch.Tensor) -> None:
        """Hook function called during forward pass."""
        self.call_count += 1
        
        # Skip if update frequency doesn't match
        if self.call_count % self.config.update_frequency != 0:
            return
        
        # Get module name (this would be set by the HooksHandler)
        module_name = getattr(module, '_hook_name', str(type(module).__name__))
        
        # Handle input (take first tensor if tuple)
        input_tensor = input[0] if isinstance(input, tuple) and len(input) > 0 else input
        if not isinstance(input_tensor, torch.Tensor):
            return
        
        # Handle output
        if not isinstance(output, torch.Tensor):
            return
        
        # Update statistics
        self.collector.update_statistics(module_name, input_tensor, output)
    
    def get_statistics(self, module_name: str) -> Optional[ModuleStatistics]:
        """Get statistics for a specific module."""
        return self.collector.get_module_statistics(module_name)
    
    def get_all_statistics(self) -> Dict[str, ModuleStatistics]:
        """Get statistics for all modules."""
        return self.collector.get_all_statistics()
    
    def export_statistics(self, format: str = "json") -> Union[Dict, str]:
        """Export statistics in the specified format."""
        return self.collector.export_statistics(format)
    
    def reset_statistics(self, module_name: Optional[str] = None) -> None:
        """Reset statistics for a specific module or all modules."""
        self.collector.clear_statistics(module_name)
        if module_name is None:
            self.call_count = 0