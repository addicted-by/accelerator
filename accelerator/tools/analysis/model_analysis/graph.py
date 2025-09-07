from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch.fx import GraphModule, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


@dataclass
class NodeSpec:
    """Specifications for a node in an FX graph."""

    name: str
    op: str
    target: Any
    target_type: Optional[str]
    shape: Optional[Tuple[int, ...]]
    dtype: Optional[torch.dtype]


def _extract_tensor_meta(node: torch.fx.Node) -> Tuple[Optional[Tuple[int, ...]], Optional[torch.dtype]]:
    """Helper to extract shape and dtype metadata from a node."""

    meta = node.meta.get("tensor_meta")
    if meta is None:
        return None, None
    return tuple(meta.shape), meta.dtype


def trace_model(
    model: torch.nn.Module, example_inputs: Optional[Iterable[torch.Tensor]] = None
) -> Tuple[GraphModule, List[NodeSpec]]:
    """Symbolically trace ``model`` and collect node metadata.

    Parameters
    ----------
    model:
        The ``torch.nn.Module`` to be traced.
    example_inputs:
        Optional iterable of example tensors used for meta shape propagation. If
        provided, tensors are converted to ``device='meta'`` and propagated
        through the graph to capture shape and dtype information.

    Returns
    -------
    GraphModule
        The traced ``GraphModule``.
    list[NodeSpec]
        Registry describing each node in the traced graph.
    """

    gm = symbolic_trace(model)

    if example_inputs is not None:
        # Move parameters and buffers to meta device so that shape propagation
        # does not allocate real memory or trigger device mismatches.
        gm.to(device="meta")
        meta_args = [t.to(device="meta") for t in example_inputs]
        ShapeProp(gm).propagate(*meta_args)

    registry: List[NodeSpec] = []
    for node in gm.graph.nodes:
        shape, dtype = _extract_tensor_meta(node)

        if node.op == "call_module":
            target_type = gm.get_submodule(str(node.target)).__class__.__name__
        elif node.op == "call_function":
            target_type = getattr(node.target, "__name__", str(node.target))
        elif node.op == "call_method":
            target_type = str(node.target)
        else:
            target_type = None

        registry.append(
            NodeSpec(
                name=node.name,
                op=node.op,
                target=node.target,
                target_type=target_type,
                shape=shape,
                dtype=dtype,
            )
        )

    return gm, registry

