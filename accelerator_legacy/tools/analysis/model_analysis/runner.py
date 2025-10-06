from __future__ import annotations

from functools import partial
from typing import Any, List

import torch
from torch.fx import GraphModule, Interpreter, Node


class NodeInterpreter(Interpreter):
    """Interpreter that routes tensors to a ``StatsRouter``.

    Each node's inputs and outputs are normalized to a list of tensors and
    dispatched to the provided ``stats_router`` for statistics collection.
    Backward statistics are captured via autograd hooks registered on the
    forward outputs.
    """

    def __init__(self, gm: GraphModule, stats_router: Any) -> None:
        super().__init__(gm)
        self.stats_router = stats_router

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _tensor_list(obj: Any) -> List[torch.Tensor]:
        """Recursively collect tensors from ``obj`` into a list."""

        tensors: List[torch.Tensor] = []

        def collect(x: Any) -> None:
            if isinstance(x, torch.Tensor):
                tensors.append(x)
            elif isinstance(x, (list, tuple, set)):
                for v in x:
                    collect(v)
            elif isinstance(x, dict):
                for v in x.values():
                    collect(v)

        collect(obj)
        return tensors

    def _dispatch(self, method: str, node: Node, tensors: List[torch.Tensor]) -> None:
        """Send ``tensors`` to the stats router for ``method`` stage."""

        router = self.stats_router
        if router is None:
            return
        if hasattr(router, method):
            getattr(router, method)(node, tensors)
        elif callable(router):
            router(node, method, tensors)

    # ------------------------------------------------------------------
    # FX Interpreter overrides
    # ------------------------------------------------------------------
    def run_node(self, n: Node) -> Any:  # type: ignore[override]
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        inputs = self._tensor_list((args, kwargs))
        self._dispatch("forward_pre", n, inputs)

        result = getattr(self, n.op)(n.target, args, kwargs)

        outputs = self._tensor_list(result)
        self._dispatch("forward_post", n, outputs)

        for t in outputs:
            if t.requires_grad:
                t.register_hook(partial(self._backward_hook, n))

        return result

    # ------------------------------------------------------------------
    # Autograd hook
    # ------------------------------------------------------------------
    def _backward_hook(self, node: Node, grad: torch.Tensor) -> torch.Tensor:
        grads = self._tensor_list(grad)
        self._dispatch("backward", node, grads)
        return grad

