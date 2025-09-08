from __future__ import annotations

from fnmatch import fnmatch
from typing import Iterable, Optional, Set

from accelerator.tools.analysis.stats import TensorStatsCollector


class StatsRouter:
    """Route tensors from ``NodeInterpreter`` to a ``TensorStatsCollector``.

    Parameters
    ----------
    collector:
        Instance of :class:`TensorStatsCollector` receiving statistics updates.
    nodes:
        Optional iterable of node names to consider. If ``None`` all nodes are
        considered.
    filter_layers:
        Optional patterns that must match a node name for statistics to be
        collected.
    exclude_layers:
        Optional patterns used to exclude matching node names from statistics
        collection.
    """

    def __init__(
        self,
        collector: TensorStatsCollector,
        nodes: Optional[Iterable[str]] = None,
        filter_layers: Optional[Iterable[str]] = None,
        exclude_layers: Optional[Iterable[str]] = None,
    ) -> None:
        self.collector = collector
        self.nodes: Optional[Set[str]] = set(nodes) if nodes is not None else None
        self.filter_layers = list(filter_layers or [])
        self.exclude_layers = list(exclude_layers or [])

    # ------------------------------------------------------------------
    def _match(self, name: str) -> bool:
        if self.nodes is not None and name not in self.nodes:
            return False
        if self.filter_layers and not any(fnmatch(name, p) for p in self.filter_layers):
            return False
        if any(fnmatch(name, p) for p in self.exclude_layers):
            return False
        return True

    # ------------------------------------------------------------------
    def forward_post(self, node, tensors) -> None:
        if self._match(node.name):
            self.collector.update_activation(node.name, tensors)

    def backward(self, node, tensors) -> None:
        if self._match(node.name):
            self.collector.update_gradient(node.name, tensors)
