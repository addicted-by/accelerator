import threading
import time
from typing import Any, Dict, Iterable, Tuple

import torch
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlInit, nvmlShutdown

from accelerator.utilities.move_to_device import move_data_to_device


def _prepare_inputs(inputs: Any) -> Tuple[Any, ...]:
    if isinstance(inputs, (list, tuple)):
        return tuple(inputs)
    return (inputs,)


def measure_gpu_power(
    model: torch.nn.Module,
    dataloader: Iterable[Any],
    device: int | torch.device | str,
    interval_s: float = 0.1,
) -> Dict[str, float]:
    """Measure GPU power usage during model inference.

    Args:
        model: Model to evaluate in ``eval`` mode.
        dataloader: Iterable providing batches of inputs for the model.
        device: Device index or specification on which to run the model.
        interval_s: Time in seconds between power samples.

    Returns:
        Dictionary containing total energy (J), mean power (W), and peak power (W).
    """
    if isinstance(device, int):
        torch_device = torch.device(f"cuda:{device}")
        device_index = device
    else:
        torch_device = torch.device(device)
        device_index = torch_device.index or 0

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)

    stop_event = threading.Event()
    samples: list[float] = []

    def _sample() -> None:
        while not stop_event.is_set():
            power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            samples.append(power)
            time.sleep(interval_s)

    sampler = threading.Thread(target=_sample)
    sampler.start()

    model.to(torch_device)
    model.eval()
    start = time.perf_counter()

    with torch.no_grad():
        for batch in dataloader:
            batch = move_data_to_device(batch, torch_device)
            inputs = _prepare_inputs(batch)
            model(*inputs)

    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)

    end = time.perf_counter()
    stop_event.set()
    sampler.join()
    nvmlShutdown()

    elapsed = end - start
    mean_power = sum(samples) / len(samples) if samples else 0.0
    peak_power = max(samples) if samples else 0.0
    total_energy = mean_power * elapsed

    return {
        "total_energy": total_energy,
        "mean_power": mean_power,
        "peak_power": peak_power,
    }
