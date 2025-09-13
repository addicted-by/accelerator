import pytest
import torch
from torch.fx import symbolic_trace

from accelerator.tools.analysis.model_analysis import NodeInterpreter, StatsRouter
from accelerator.tools.analysis.stats import StatsConfig, TensorStatsCollector

try:
    from torchvision.models import resnet18  # type: ignore
except Exception:  # pragma: no cover
    resnet18 = None


@pytest.mark.skipif(resnet18 is None, reason="torchvision is not installed")
def test_model_analysis_resnet18_populates_stats():
    model = resnet18(weights=None)
    gm = symbolic_trace(model)

    cfg = StatsConfig(channel_dim=1)
    collector = TensorStatsCollector(cfg, cfg)
    router = StatsRouter(collector)
    interpreter = NodeInterpreter(gm, router)

    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = interpreter.run(x)
    out.sum().backward()

    _, activations, gradients = collector.compute()

    for name in ["conv1", "fc"]:
        assert name in activations
        assert name in gradients
        assert "mean" in activations[name]
        assert "mean" in gradients[name]
        assert activations[name]["mean"].numel() > 0
        assert gradients[name]["mean"].numel() > 0
