from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Type

import torch


# ---------------------------------------------------------------------------
# Base collectors
# ---------------------------------------------------------------------------


class _BaseTensorCollector:
    """Base class for simple tensor statistics collectors."""

    name: str

    def __init__(self, channel_dim: int) -> None:
        self.channel_dim = channel_dim

    def update(self, tensor: torch.Tensor) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def compute(self) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class MeanTensorCollector(_BaseTensorCollector):
    """Collector computing per-channel mean values."""

    name = "mean"

    def __init__(self, channel_dim: int) -> None:
        super().__init__(channel_dim)
        self._sum: torch.Tensor | None = None
        self._count: int = 0

    def update(self, tensor: torch.Tensor) -> None:  # pragma: no cover - simple math
        dims = tuple(d for d in range(tensor.dim()) if d != self.channel_dim)
        summed = tensor.float().sum(dim=dims)
        count = tensor.numel() // tensor.size(self.channel_dim)
        if self._sum is None:
            self._sum = summed
        else:
            self._sum = self._sum + summed
        self._count += count

    def compute(self) -> torch.Tensor:  # pragma: no cover - simple math
        if self._sum is None or self._count == 0:
            return torch.tensor(0.0)
        return self._sum / self._count


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StatsConfig:
    """Configuration for tensor statistics collection.

    Parameters
    ----------
    channel_dim:
        Dimension treated as channel dimension. Reduction is performed over all
        other dimensions.
    collectors:
        Iterable of collector classes used for gathering statistics. Each class
        must derive from ``_BaseTensorCollector``.
    """

    channel_dim: int = 0
    collectors: Tuple[Type[_BaseTensorCollector], ...] = (MeanTensorCollector,)


# ---------------------------------------------------------------------------
# Tensor stats collection
# ---------------------------------------------------------------------------


class TensorStatsCollector:
    """Collects activation and gradient statistics per FX node."""

    def __init__(
        self,
        activation_config: StatsConfig,
        gradient_config: StatsConfig,
        activation_configs: Dict[str, StatsConfig] | None = None,
        gradient_configs: Dict[str, StatsConfig] | None = None,
    ) -> None:
        self.default_activation_config = activation_config
        self.default_gradient_config = gradient_config
        self.activation_configs = activation_configs or {}
        self.gradient_configs = gradient_configs or {}

        self.activation_collectors: Dict[str, List[_BaseTensorCollector]] = {}
        self.gradient_collectors: Dict[str, List[_BaseTensorCollector]] = {}

        for node, cfg in self.activation_configs.items():
            self.activation_collectors[node] = [c(channel_dim=cfg.channel_dim) for c in cfg.collectors]
        for node, cfg in self.gradient_configs.items():
            self.gradient_collectors[node] = [c(channel_dim=cfg.channel_dim) for c in cfg.collectors]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_collectors(
        self,
        node: str,
        configs: Dict[str, StatsConfig],
        store: Dict[str, List[_BaseTensorCollector]],
        default_config: StatsConfig,
    ) -> List[_BaseTensorCollector]:
        if node not in store:
            cfg = configs.get(node, default_config)
            store[node] = [c(channel_dim=cfg.channel_dim) for c in cfg.collectors]
        return store[node]

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------
    def update_activation(self, node: str, tensors: Iterable[torch.Tensor]) -> None:
        collectors = self._get_collectors(
            node, self.activation_configs, self.activation_collectors, self.default_activation_config
        )
        for t in tensors:
            for c in collectors:
                c.update(t)

    def update_gradient(self, node: str, tensors: Iterable[torch.Tensor]) -> None:
        collectors = self._get_collectors(
            node, self.gradient_configs, self.gradient_collectors, self.default_gradient_config
        )
        for t in tensors:
            for c in collectors:
                c.update(t)

    # ------------------------------------------------------------------
    # Compute results
    # ------------------------------------------------------------------
    def compute(self) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
        activations = {
            node: {collector.name: collector.compute() for collector in collectors}
            for node, collectors in self.activation_collectors.items()
        }
        gradients = {
            node: {collector.name: collector.compute() for collector in collectors}
            for node, collectors in self.gradient_collectors.items()
        }
        return activations, gradients
