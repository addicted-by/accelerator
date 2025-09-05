# Activation Statistics Hooks

This module provides comprehensive activation statistics collection for PyTorch neural networks. It's designed to be easy to use, memory-efficient, and integrate seamlessly with your existing training and inference workflows.

## Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from accelerator.hooks import ActivationStatisticsHook, StatisticsConfig

# Create your model
model = YourModel()

# Configure statistics collection
config = StatisticsConfig(
    collect_mean=True,
    collect_std=True,
    collect_min_max=True,
    collect_percentiles=[95, 99]
)

# Create statistics hook
stats_hook = ActivationStatisticsHook(config)

# Register hooks on specific layers
hooks = []
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        module._hook_name = name  # Set name for identification
        hook_handle = module.register_forward_hook(stats_hook)
        hooks.append(hook_handle)

# Run your model
model.eval()
with torch.no_grad():
    for batch in dataloader:
        output = model(batch)

# Get statistics
all_stats = stats_hook.get_all_statistics()
for module_name, stats in all_stats.items():
    print(f"{module_name}: mean={stats.output_stats.mean:.4f}, std={stats.output_stats.std:.4f}")

# Export statistics
stats_data = stats_hook.export_statistics(format="json")

# Clean up
for hook in hooks:
    hook.remove()
```

## Configuration Options

### StatisticsConfig

```python
@dataclass
class StatisticsConfig:
    # Basic statistics
    collect_mean: bool = True          # Collect mean values
    collect_std: bool = True           # Collect standard deviation
    collect_min_max: bool = True       # Collect min/max values
    collect_percentiles: List[float] = [95, 99]  # Percentiles to compute
    
    # Advanced statistics
    collect_histogram: bool = False    # Collect histogram data
    histogram_bins: int = 100         # Number of histogram bins
    collect_sparsity: bool = False    # Compute sparsity ratio
    
    # Performance settings
    update_frequency: int = 1         # Update every N forward passes
    max_samples: Optional[int] = 1000 # Limit samples for memory efficiency
    
    # Memory management
    max_memory_mb: Optional[float] = None  # Memory limit in MB
```

## Use Cases

### 1. Quantization Calibration

```python
# Configure for quantization calibration
config = StatisticsConfig(
    collect_min_max=True,      # Essential for quantization
    collect_percentiles=[99.9], # For outlier handling
    collect_mean=False,        # Not needed for quantization
    collect_std=False,         # Not needed for quantization
    max_samples=5000          # Limit memory usage
)

stats_hook = ActivationStatisticsHook(config)

# Register on all layers
for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # Leaf modules only
        module._hook_name = name
        module.register_forward_hook(stats_hook)

# Run calibration
model.eval()
with torch.no_grad():
    for batch in calibration_loader:
        model(batch)

# Export quantization data
quant_data = stats_hook.export_statistics(format="quantization")
```

### 2. Training Monitoring

```python
# Configure for training monitoring
config = StatisticsConfig(
    collect_mean=True,
    collect_std=True,
    collect_sparsity=True,
    update_frequency=10,  # Update every 10 forward passes
    max_samples=1000
)

stats_hook = ActivationStatisticsHook(config)

# Register on key layers
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        module._hook_name = name
        module.register_forward_hook(stats_hook)

# Training loop with monitoring
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # ... training code ...
        
        # Check for issues periodically
        if batch_idx % 100 == 0:
            all_stats = stats_hook.get_all_statistics()
            for module_name, stats in all_stats.items():
                # Check for dead neurons
                if stats.output_stats.sparsity_ratio > 0.9:
                    print(f"Warning: {module_name} has high sparsity")
                
                # Check for vanishing activations
                if stats.output_stats.std < 0.01:
                    print(f"Warning: {module_name} has low variance")
```

### 3. Model Analysis

```python
# Configure for comprehensive analysis
config = StatisticsConfig(
    collect_mean=True,
    collect_std=True,
    collect_min_max=True,
    collect_percentiles=[1, 5, 25, 50, 75, 95, 99],
    collect_sparsity=True,
    collect_histogram=True,
    histogram_bins=50
)

stats_hook = ActivationStatisticsHook(config)

# Register on all layers
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        module._hook_name = name
        module.register_forward_hook(stats_hook)

# Run analysis
model.eval()
with torch.no_grad():
    for batch in analysis_loader:
        model(batch)

# Comprehensive analysis
all_stats = stats_hook.get_all_statistics()

# Find problematic layers
for module_name, stats in all_stats.items():
    output_stats = stats.output_stats
    
    # Check activation range
    if output_stats.max - output_stats.min < 0.1:
        print(f"{module_name}: Very narrow activation range")
    
    # Check for saturation
    if output_stats.percentiles.get(99, 0) == output_stats.max:
        print(f"{module_name}: Possible activation saturation")
    
    # Check distribution
    if abs(output_stats.mean) > 2 * output_stats.std:
        print(f"{module_name}: Skewed activation distribution")
```

## Advanced Usage

### Custom Statistics Collector

```python
from accelerator.hooks import StatisticsCollector

# Create custom collector
collector = StatisticsCollector(config)

# Manual statistics collection during custom forward pass
def custom_forward(model, x):
    # Layer 1
    conv1_out = model.conv1(x)
    collector.update_statistics("conv1", x, conv1_out)
    
    # Layer 2
    relu_out = torch.relu(conv1_out)
    collector.update_statistics("relu", conv1_out, relu_out)
    
    # Continue for other layers...
    return final_output

# Use custom forward pass
for batch in dataloader:
    output = custom_forward(model, batch)

# Get statistics
stats = collector.get_all_statistics()
```

### Memory-Efficient Collection

```python
# Configure for large models with memory constraints
config = StatisticsConfig(
    collect_mean=True,
    collect_std=True,
    collect_min_max=True,
    collect_percentiles=[95],  # Fewer percentiles
    max_samples=100,          # Very limited samples
    update_frequency=5,       # Less frequent updates
    max_memory_mb=50         # Memory limit
)

stats_hook = ActivationStatisticsHook(config)

# Register selectively on important layers only
important_layers = ["conv1", "conv2", "fc1"]
for name, module in model.named_modules():
    if name in important_layers:
        module._hook_name = name
        module.register_forward_hook(stats_hook)
```

## Export Formats

### JSON Export

```python
# Export as JSON
stats_data = stats_hook.export_statistics(format="json")

# Structure:
{
    "module_name": {
        "total_samples": 100,
        "last_updated": "2024-01-01T12:00:00",
        "input_stats": {
            "mean": 0.1234,
            "std": 0.5678,
            "min": -2.0,
            "max": 3.0,
            "percentiles": {"95": 1.5, "99": 2.0},
            "sparsity_ratio": 0.1
        },
        "output_stats": { ... }
    }
}
```

### Quantization Export

```python
# Export for quantization tools
quant_data = stats_hook.export_statistics(format="quantization")

# Structure:
{
    "module_name": {
        "min": -2.0,
        "max": 3.0,
        "percentiles": {"99.9": 2.5}
    }
}
```

## Performance Considerations

1. **Update Frequency**: Use `update_frequency > 1` to reduce overhead
2. **Sample Limits**: Set `max_samples` to limit memory usage
3. **Selective Collection**: Only collect statistics you need
4. **Memory Limits**: Set `max_memory_mb` for automatic cleanup
5. **Selective Registration**: Only register hooks on important layers

## Best Practices

1. **Always clean up hooks** after use to prevent memory leaks
2. **Use appropriate sample limits** for your memory constraints
3. **Monitor memory usage** when collecting statistics on large models
4. **Export statistics regularly** to avoid data loss
5. **Use selective collection** for production environments

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Reduce `max_samples` or increase `update_frequency`
2. **Performance Impact**: Use selective layer registration and higher `update_frequency`
3. **Missing Statistics**: Ensure `_hook_name` is set on modules
4. **Export Errors**: Check that statistics have been collected before export

### Debug Mode

```python
# Enable debug information
config = StatisticsConfig(collect_mean=True)
stats_hook = ActivationStatisticsHook(config)

# Check collection status
print(f"Total modules with statistics: {len(stats_hook.get_all_statistics())}")
print(f"Hook call count: {stats_hook.call_count}")

# Verify specific module
stats = stats_hook.get_statistics("conv1")
if stats is None:
    print("No statistics collected for conv1")
else:
    print(f"Conv1 samples: {stats.total_samples}")
```