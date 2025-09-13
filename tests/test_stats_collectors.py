import torch

from accelerator.tools.analysis.stats.collector import StatsConfig, TensorStatsCollector


def test_per_layer_channel_dim_configs():
    default_cfg = StatsConfig(channel_dim=0)
    layer1_cfg = StatsConfig(channel_dim=0)
    layer2_cfg = StatsConfig(channel_dim=1)

    activation_configs = {"layer1": layer1_cfg, "layer2": layer2_cfg}
    gradient_configs = {"layer1": layer1_cfg, "layer2": layer2_cfg}

    collector = TensorStatsCollector(
        activation_config=default_cfg,
        gradient_config=default_cfg,
        activation_configs=activation_configs,
        gradient_configs=gradient_configs,
    )

    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    collector.update_activation("layer1", [tensor])
    collector.update_activation("layer2", [tensor])

    grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    collector.update_gradient("layer1", [grad])
    collector.update_gradient("layer2", [grad])

    input_activations, activations, gradients = collector.compute()

    assert input_activations == {}

    assert torch.allclose(activations["layer1"]["mean"], torch.tensor([1.5, 3.5]))
    assert torch.allclose(activations["layer2"]["mean"], torch.tensor([2.0, 3.0]))

    assert torch.allclose(gradients["layer1"]["mean"], torch.tensor([1.5, 3.5]))
    assert torch.allclose(gradients["layer2"]["mean"], torch.tensor([2.0, 3.0]))

