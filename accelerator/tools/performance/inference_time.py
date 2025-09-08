import time
from typing import Any, Dict, Tuple

import torch
from torch.profiler import ProfilerActivity, profile

from accelerator.utilities.move_to_device import move_data_to_device


def _prepare_inputs(inputs: Any) -> Tuple[Any, ...]:
    if isinstance(inputs, (list, tuple)):
        return tuple(inputs)
    return (inputs,)


def measure_inference_time(
    model: torch.nn.Module,
    inputs: Any,
    warmup: int = 5,
    runs: int = 50,
    device: torch.device | str | None = None,
) -> Dict[str, float]:
    """Measure inference latency for a model.

    Args:
        model: Model to evaluate in ``eval`` mode.
        inputs: Inputs to pass to the model. Can be any structure accepted by
            :func:`move_data_to_device`.
        warmup: Number of warmup iterations to run before timing.
        runs: Number of timed runs to average over.
        device: Device on which to run the model. If ``None``, uses the model's
            current device.

    Returns:
        Dictionary containing average latency, variance, and percentile metrics
        (p50, p90, p95, p99).
    """
    if device is not None:
        device = torch.device(device)
        model.to(device)
    else:
        device = next(model.parameters()).device

    model.eval()
    inputs = move_data_to_device(inputs, device)
    inputs = _prepare_inputs(inputs)

    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)

        timings: list[float] = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

    times = torch.tensor(timings)
    metrics = {
        "average": times.mean().item(),
        "variance": times.var(unbiased=False).item(),
        "p50": times.quantile(0.5).item(),
        "p90": times.quantile(0.9).item(),
        "p95": times.quantile(0.95).item(),
        "p99": times.quantile(0.99).item(),
    }
    return metrics


def measure_per_node_inference_time(
    model: torch.nn.Module,
    inputs: Any,
    warmup: int = 5,
    runs: int = 50,
    device: torch.device | str | None = None,
) -> Dict[str, Dict[str, float]]:
    """Profile per-operator execution time for a model.

    Uses :mod:`torch.profiler` to capture self CPU and CUDA times for each
    operator executed during inference. Times are reported in seconds and
    averaged over ``runs`` iterations.

    Args:
        model: Model to evaluate in ``eval`` mode.
        inputs: Inputs to pass to the model. Can be any structure accepted by
            :func:`move_data_to_device`.
        warmup: Number of warmup iterations to run before profiling.
        runs: Number of profiled runs to average over.
        device: Device on which to run the model. If ``None``, uses the model's
            current device.

    Returns:
        Dictionary mapping operator names to dictionaries containing averaged
        CPU and CUDA self times in seconds.
    """

    if device is not None:
        device = torch.device(device)
        model.to(device)
    else:
        device = next(model.parameters()).device

    model.eval()
    inputs = move_data_to_device(inputs, device)
    inputs = _prepare_inputs(inputs)

    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=True) as prof:
            for _ in range(runs):
                model(*inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()

    metrics: Dict[str, Dict[str, float]] = {}
    for evt in prof.key_averages():
        entry: Dict[str, float] = {}
        if evt.self_cpu_time_total > 0:
            entry["cpu_time_avg"] = (evt.self_cpu_time_total / runs) / 1e6
        if hasattr(evt, "self_cuda_time_total") and evt.self_cuda_time_total > 0:
            entry["cuda_time_avg"] = (evt.self_cuda_time_total / runs) / 1e6
        if entry:
            metrics[evt.key] = entry

    return metrics
