import torch
import pytest
from torch.fx import GraphModule

from accelerator.tools.analysis.model_analysis import NodeSpec, trace_model

class Simple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x):  # pragma: no cover - executed via tracing
        return torch.relu(self.linear(x))


def test_trace_model_returns_graph_and_registry():
    model = Simple()
    gm, registry = trace_model(model, (torch.randn(2, 3),))

    assert isinstance(gm, GraphModule)
    assert all(isinstance(n, NodeSpec) for n in registry)
    assert len(registry) == len(list(gm.graph.nodes))

    lin_spec = next(n for n in registry if n.name == "linear")
    assert lin_spec.shape == (2, 4)
    assert lin_spec.dtype == torch.float32
    assert lin_spec.target_type == "Linear"

    relu_spec = next(n for n in registry if n.target_type == "relu")
    assert relu_spec.op == "call_function"
    input_spec = next(n for n in registry if n.op == "placeholder")
    assert input_spec.shape == (2, 3)
    assert input_spec.dtype == torch.float32


def test_trace_model_handles_resnet18():
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None)
    gm, registry = trace_model(model, (torch.randn(1, 3, 224, 224),))

    fc_spec = next(n for n in registry if n.name == "fc")
    assert fc_spec.shape == (1, 1000)
    assert fc_spec.dtype == torch.float32


def test_traced_module_runs_after_shape_prop():
    model = Simple()
    inputs = torch.randn(2, 3)
    gm, _ = trace_model(model, (inputs,))

    # Ensure the traced module can still execute with real inputs and produces
    # the same results as the original model.
    torch.testing.assert_close(gm(inputs), model(inputs))
