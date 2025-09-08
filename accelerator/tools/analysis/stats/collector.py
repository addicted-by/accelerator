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


def _flatten_tensor(tensor: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """Return tensor flattened to (C, N) in float32 for stable accumulation."""
    t = tensor.float().transpose(channel_dim, 0).contiguous()
    return t.view(t.size(0), -1)


class _WelfordCollector(_BaseTensorCollector):
    """Base collector implementing Welford's online algorithm.

    This collector keeps track of higher order moments for each channel, which
    allows subclasses to compute variance, skewness and kurtosis without storing
    the whole tensor."""

    def __init__(self, channel_dim: int, order: int = 2) -> None:
        super().__init__(channel_dim)
        self.order = order
        self._count: int = 0
        self._mean: torch.Tensor | None = None
        self._M2: torch.Tensor | None = None
        self._M3: torch.Tensor | None = None
        self._M4: torch.Tensor | None = None

    def update(self, tensor: torch.Tensor) -> None:  # pragma: no cover - math
        flat = _flatten_tensor(tensor, self.channel_dim)
        n_b = flat.size(1)
        mean_b = flat.mean(dim=1)
        centered = flat - mean_b[:, None]
        M2_b = (centered ** 2).sum(dim=1)
        M3_b = (centered ** 3).sum(dim=1) if self.order > 2 else None
        M4_b = (centered ** 4).sum(dim=1) if self.order > 3 else None

        if self._count == 0:
            self._count = n_b
            self._mean = mean_b
            self._M2 = M2_b
            if self.order > 2:
                self._M3 = M3_b
            if self.order > 3:
                self._M4 = M4_b
            return

        # combine existing stats with new batch statistics
        n_a = self._count
        mean_a = self._mean  # type: ignore[arg-type]
        M2_a = self._M2  # type: ignore[arg-type]
        n = n_a + n_b
        delta = mean_b - mean_a
        self._mean = mean_a + delta * n_b / n
        self._M2 = M2_a + M2_b + delta**2 * n_a * n_b / n

        if self.order > 2:
            M3_a = self._M3  # type: ignore[arg-type]
            self._M3 = (
                M3_a
                + M3_b
                + delta**3 * n_a * n_b * (n_a - n_b) / (n**2)
                + 3 * delta * (n_a * M2_b - n_b * M2_a) / n
            )
        if self.order > 3:
            M4_a = self._M4  # type: ignore[arg-type]
            M3_a = self._M3  # type: ignore[arg-type]
            self._M4 = (
                M4_a
                + M4_b
                + delta**4 * n_a * n_b * (n_a**2 - n_a * n_b + n_b**2) / (n**3)
                + 6 * delta**2 * (n_a**2 * M2_b + n_b**2 * M2_a) / (n**2)
                + 4 * delta * (n_a * M3_b - n_b * M3_a) / n
            )

        self._count = n


class VarianceTensorCollector(_WelfordCollector):
    """Collector computing per-channel variance using Welford's algorithm."""

    name = "var"

    def __init__(self, channel_dim: int) -> None:
        super().__init__(channel_dim, order=2)

    def compute(self) -> torch.Tensor:  # pragma: no cover - simple math
        if self._count < 2 or self._M2 is None:
            return torch.tensor(0.0)
        return self._M2 / (self._count - 1)


class SkewnessTensorCollector(_WelfordCollector):
    """Collector computing per-channel skewness."""

    name = "skewness"

    def __init__(self, channel_dim: int) -> None:
        super().__init__(channel_dim, order=3)

    def compute(self) -> torch.Tensor:  # pragma: no cover - simple math
        if self._count < 2 or self._M2 is None or self._M3 is None:
            return torch.tensor(0.0)
        m2 = self._M2 / (self._count - 1)
        m3 = self._M3 / self._count
        return (torch.sqrt(torch.tensor(self._count, dtype=torch.float32)) * m3) / (m2 ** 1.5)


class KurtosisTensorCollector(_WelfordCollector):
    """Collector computing per-channel excess kurtosis."""

    name = "kurtosis"

    def __init__(self, channel_dim: int) -> None:
        super().__init__(channel_dim, order=4)

    def compute(self) -> torch.Tensor:  # pragma: no cover - simple math
        if self._count < 2 or self._M2 is None or self._M4 is None:
            return torch.tensor(0.0)
        m2 = self._M2 / (self._count - 1)
        m4 = self._M4 / self._count
        return (self._count * m4) / (m2 ** 2) - 3


class MinTensorCollector(_BaseTensorCollector):
    """Collector tracking per-channel minimum values."""

    name = "min"

    def __init__(self, channel_dim: int) -> None:
        super().__init__(channel_dim)
        self._min: torch.Tensor | None = None

    def update(self, tensor: torch.Tensor) -> None:  # pragma: no cover - math
        flat = _flatten_tensor(tensor, self.channel_dim)
        cur = flat.min(dim=1).values
        if self._min is None:
            self._min = cur
        else:
            self._min = torch.minimum(self._min, cur)

    def compute(self) -> torch.Tensor:  # pragma: no cover - simple math
        return self._min if self._min is not None else torch.tensor(0.0)


class MaxTensorCollector(_BaseTensorCollector):
    """Collector tracking per-channel maximum values."""

    name = "max"

    def __init__(self, channel_dim: int) -> None:
        super().__init__(channel_dim)
        self._max: torch.Tensor | None = None

    def update(self, tensor: torch.Tensor) -> None:  # pragma: no cover - math
        flat = _flatten_tensor(tensor, self.channel_dim)
        cur = flat.max(dim=1).values
        if self._max is None:
            self._max = cur
        else:
            self._max = torch.maximum(self._max, cur)

    def compute(self) -> torch.Tensor:  # pragma: no cover - simple math
        return self._max if self._max is not None else torch.tensor(0.0)


class SparsityTensorCollector(_BaseTensorCollector):
    """Collector tracking ratio of zero elements per channel."""

    name = "sparsity"

    def __init__(self, channel_dim: int) -> None:
        super().__init__(channel_dim)
        self._zero: torch.Tensor | None = None
        self._count: int = 0

    def update(self, tensor: torch.Tensor) -> None:  # pragma: no cover - math
        flat = _flatten_tensor(tensor, self.channel_dim)
        zeros = (flat == 0).sum(dim=1).float()
        n = flat.size(1)
        if self._zero is None:
            self._zero = zeros
        else:
            self._zero += zeros
        self._count += n

    def compute(self) -> torch.Tensor:  # pragma: no cover - simple math
        if self._zero is None or self._count == 0:
            return torch.tensor(0.0)
        return self._zero / self._count


class L1NormTensorCollector(_BaseTensorCollector):
    """Collector computing per-channel L1 norm."""

    name = "l1_norm"

    def __init__(self, channel_dim: int) -> None:
        super().__init__(channel_dim)
        self._sum: torch.Tensor | None = None

    def update(self, tensor: torch.Tensor) -> None:  # pragma: no cover - math
        flat = _flatten_tensor(tensor, self.channel_dim)
        val = flat.abs().sum(dim=1)
        if self._sum is None:
            self._sum = val
        else:
            self._sum += val

    def compute(self) -> torch.Tensor:  # pragma: no cover - simple math
        return self._sum if self._sum is not None else torch.tensor(0.0)


class L2NormTensorCollector(_BaseTensorCollector):
    """Collector computing per-channel L2 norm."""

    name = "l2_norm"

    def __init__(self, channel_dim: int) -> None:
        super().__init__(channel_dim)
        self._sum_sq: torch.Tensor | None = None

    def update(self, tensor: torch.Tensor) -> None:  # pragma: no cover - math
        flat = _flatten_tensor(tensor, self.channel_dim)
        val = (flat ** 2).sum(dim=1)
        if self._sum_sq is None:
            self._sum_sq = val
        else:
            self._sum_sq += val

    def compute(self) -> torch.Tensor:  # pragma: no cover - simple math
        if self._sum_sq is None:
            return torch.tensor(0.0)
        return torch.sqrt(self._sum_sq)


class _P2Quantile:
    """P\u00b2 quantile estimator for a single quantile."""

    def __init__(self, quantile: float) -> None:
        self.quantile = quantile
        self.n = 0
        self.initial: List[float] = []
        self.q = torch.zeros(5, dtype=torch.float32)
        self.np = torch.zeros(5, dtype=torch.float32)
        self.np_des = torch.zeros(5, dtype=torch.float32)
        self.dn = torch.zeros(5, dtype=torch.float32)

    def update(self, x: float) -> None:
        if self.n < 5:
            self.initial.append(x)
            self.n += 1
            if self.n == 5:
                self.initial.sort()
                self.q = torch.tensor(self.initial, dtype=torch.float32)
                self.np = torch.arange(1, 6, dtype=torch.float32)
                p = self.quantile
                self.np_des = torch.tensor([1, 1 + 2 * p, 1 + 4 * p, 3 + 2 * p, 5], dtype=torch.float32)
                self.dn = torch.tensor([0, p / 2, p, (1 + p) / 2, 1], dtype=torch.float32)
            return

        k = torch.searchsorted(self.q, torch.tensor([x])).item()
        if k == 0:
            self.q[0] = x
            k = 1
        elif k == 5:
            self.q[4] = x
            k = 4
        self.np[k:5] += 1
        self.n += 1
        self.np_des += self.dn
        for i in range(1, 4):
            d = self.np_des[i] - self.np[i]
            if (d >= 1 and self.np[i + 1] - self.np[i] > 1) or (d <= -1 and self.np[i - 1] - self.np[i] < -1):
                d = torch.sign(d)
                qs = self.q[i] + d * (
                    (self.np[i] - self.np[i - 1] + d) * (self.q[i + 1] - self.q[i]) / (self.np[i + 1] - self.np[i])
                    + (self.np[i + 1] - self.np[i] - d) * (self.q[i] - self.q[i - 1]) / (self.np[i] - self.np[i - 1])
                ) / (self.np[i + 1] - self.np[i - 1])
                if self.q[i - 1] < qs < self.q[i + 1]:
                    self.q[i] = qs
                else:
                    di = int(d.item())
                    self.q[i] += d * (self.q[i + di] - self.q[i]) / (self.np[i + di] - self.np[i])
                self.np[i] += d

    def result(self) -> float:
        if self.n < 5:
            if not self.initial:
                return 0.0
            self.initial.sort()
            idx = int(round(self.quantile * (len(self.initial) - 1)))
            return float(self.initial[idx])
        return float(self.q[2])


class PercentileTensorCollector(_BaseTensorCollector):
    """Collector estimating percentiles using the P\u00b2 algorithm."""

    name = "percentiles"
    percentiles: Tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)

    def __init__(self, channel_dim: int) -> None:
        super().__init__(channel_dim)
        self.estimators: List[List[_P2Quantile]] | None = None

    def _ensure_estimators(self, channels: int) -> None:
        if self.estimators is None:
            self.estimators = [
                [_P2Quantile(p) for p in self.percentiles] for _ in range(channels)
            ]

    def update(self, tensor: torch.Tensor) -> None:  # pragma: no cover - loops
        flat = _flatten_tensor(tensor, self.channel_dim)
        self._ensure_estimators(flat.size(0))
        for c in range(flat.size(0)):
            for v in flat[c].tolist():
                for est in self.estimators[c]:
                    est.update(float(v))

    def compute(self) -> torch.Tensor:  # pragma: no cover - simple math
        if not self.estimators:
            return torch.tensor(0.0)
        res = torch.tensor(
            [[est.result() for est in channel] for channel in self.estimators],
            dtype=torch.float32,
        )
        return res


class HistogramTensorCollector(_BaseTensorCollector):
    """Collector building per-channel histograms with dynamic ranges."""

    name = "histogram"
    bins: int = 10

    def __init__(self, channel_dim: int) -> None:
        super().__init__(channel_dim)
        self.hist: torch.Tensor | None = None
        self.min: torch.Tensor | None = None
        self.max: torch.Tensor | None = None

    def _rebin(self, new_min: torch.Tensor, new_max: torch.Tensor) -> None:
        assert self.hist is not None and self.min is not None and self.max is not None
        old_hist = self.hist
        new_hist = torch.zeros_like(old_hist)
        for c in range(old_hist.size(0)):
            centers = torch.linspace(self.min[c], self.max[c], self.bins, dtype=torch.float32)
            idx = ((centers - new_min[c]) / (new_max[c] - new_min[c]) * self.bins).floor().clamp(0, self.bins - 1).long()
            for i, count in enumerate(old_hist[c]):
                new_hist[c, idx[i]] += count
        self.hist = new_hist
        self.min = new_min
        self.max = new_max

    def update(self, tensor: torch.Tensor) -> None:  # pragma: no cover - loops
        flat = _flatten_tensor(tensor, self.channel_dim)
        min_val = flat.min(dim=1).values
        max_val = flat.max(dim=1).values
        if self.hist is None:
            self.hist = torch.zeros(flat.size(0), self.bins, dtype=torch.float32)
            self.min = min_val
            self.max = max_val
        else:
            new_min = torch.minimum(self.min, min_val)  # type: ignore[arg-type]
            new_max = torch.maximum(self.max, max_val)  # type: ignore[arg-type]
            if torch.any(new_min < self.min) or torch.any(new_max > self.max):  # type: ignore[arg-type]
                self._rebin(new_min, new_max)
            self.min = new_min
            self.max = new_max
        for c in range(flat.size(0)):
            rmin = self.min[c].item()
            rmax = self.max[c].item()
            if rmin == rmax:
                rmax = rmin + 1e-6
            self.hist[c] += torch.histc(flat[c], bins=self.bins, min=rmin, max=rmax)

    def compute(self) -> torch.Tensor:  # pragma: no cover - simple math
        return self.hist if self.hist is not None else torch.tensor(0.0)


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
