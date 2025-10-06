# Tools

## Performance Utilities

The `accelerator.tools.performance` module offers helpers for measuring model inference speed.

### `measure_inference_time`

Runs a model multiple times and reports aggregated latency metrics (average, variance and percentiles).

```python
from accelerator.tools.performance import measure_inference_time
metrics = measure_inference_time(model, inputs)
print(metrics)
```

### `measure_per_node_inference_time`

Profiles each operator executed during inference and reports average self CPU and CUDA times.

```python
from accelerator.tools.performance import measure_per_node_inference_time
node_metrics = measure_per_node_inference_time(model, inputs)
print(node_metrics["aten::linear"])
```

See [examples/inference_time_example.py](../examples/inference_time_example.py) for a usage example.

### `measure_gpu_power`

Samples GPU power draw during inference to report total energy, mean, and peak power usage.

```python
from accelerator.tools.performance import measure_gpu_power
metrics = measure_gpu_power(model, dataloader, device=0)
print(metrics)
```
