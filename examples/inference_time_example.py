"""Example of benchmarking a simple model's inference performance."""

import os
import sys

import torch
import torch.nn as nn

# Ensure the repository root is on sys.path so we can import accelerator
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from accelerator.tools.performance import (
    measure_inference_time,
    measure_per_node_inference_time,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleModel()
    inputs = torch.randn(32, 10)

    metrics = measure_inference_time(model, inputs, warmup=2, runs=10, device=device)
    print("Inference time metrics:", metrics)

    node_metrics = measure_per_node_inference_time(
        model, inputs, warmup=2, runs=10, device=device
    )
    print("Per-node average self times (seconds):")
    for op_name, times in list(node_metrics.items())[:5]:
        print(f"  {op_name}: {times}")


if __name__ == "__main__":
    main()
