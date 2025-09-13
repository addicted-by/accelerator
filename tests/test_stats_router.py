import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from accelerator.tools.analysis.model_analysis import NodeInterpreter, StatsRouter
from accelerator.tools.analysis.stats.collector import StatsConfig, TensorStatsCollector


def test_router_collects_input_stats():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.lin.weight.copy_(torch.eye(2))

        def forward(self, x):
            return self.lin(x)

    model = SimpleModel()
    gm = symbolic_trace(model)
    cfg = StatsConfig(channel_dim=0)
    collector = TensorStatsCollector(cfg, cfg)
    router = StatsRouter(collector)
    interpreter = NodeInterpreter(gm, router)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    interpreter.run(x)
    input_acts, acts, _ = collector.compute()

    assert torch.allclose(input_acts["lin"]["mean"], torch.tensor([1.5, 3.5]))
