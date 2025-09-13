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
        # Record the original device placement of parameters and buffers along
        # with their values so that we can restore the ``GraphModule`` after
        # running shape propagation on ``device='meta'``.
        param_devices = {name: p.device for name, p in gm.named_parameters()}
        buffer_devices = {name: b.device for name, b in gm.named_buffers()}
        state = gm.state_dict()

        # Move parameters and buffers to the meta device to avoid allocating
        # real memory or hitting device mismatches during propagation.
        gm.to(device="meta")
        meta_args = [t.to(device="meta") for t in example_inputs]
        ShapeProp(gm).propagate(*meta_args)

        # Restore parameters and buffers to their original devices so the
        # module can be executed with real inputs.
        gm.load_state_dict(state, assign=True)
        for name, device in param_devices.items():
            if gm.get_parameter(name).device != device:
                gm.get_parameter(name).data = gm.get_parameter(name).data.to(device)
        for name, device in buffer_devices.items():
            if gm.get_buffer(name).device != device:
                gm.get_buffer(name).data = gm.get_buffer(name).data.to(device)

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

