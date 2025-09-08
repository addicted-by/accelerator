"""Example of model analysis for ResNet-18 on CIFAR-10.

This script demonstrates how to use the model analysis utilities to collect
per-layer activation and gradient statistics.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

from accelerator.tools.analysis.model_analysis import NodeInterpreter, StatsRouter, trace_model
from accelerator.tools.analysis.stats import StatsConfig, TensorStatsCollector, save_tensor_stats


def main() -> None:
    model = resnet18(num_classes=10)
    dataset = CIFAR10(root="./data", train=False, download=True, transform=ToTensor())
    loader = DataLoader(dataset, batch_size=8)

    gm, _ = trace_model(model)
    default_cfg = StatsConfig(channel_dim=1)
    collector = TensorStatsCollector(default_cfg, default_cfg)
    router = StatsRouter(collector)
    interpreter = NodeInterpreter(gm, router)

    for batch in loader:
        inputs, _ = batch
        outputs = interpreter.run(inputs)
        outputs.sum().backward()
        break  # collect statistics from a single batch for brevity

    activations, gradients = collector.compute()
    save_tensor_stats(activations, gradients, "resnet18_cifar10_stats.json")
    print("Saved statistics to resnet18_cifar10_stats.json")


if __name__ == "__main__":
    main()
