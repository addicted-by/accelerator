import torch

from accelerator.tools.analysis.stats import HistogramTensorCollector


def test_histogram_calibrates_from_sample_buffer():
    collector = HistogramTensorCollector(channel_dim=0, calibration_size=100)

    first = torch.arange(50.0).unsqueeze(0)
    collector.update(first)
    assert collector.hist is None
    assert collector._calibration_buffer is not None
    assert collector._calibration_buffer.numel() == 50

    second = torch.arange(50.0, 100.0).unsqueeze(0)
    collector.update(second)
    assert collector.hist is not None
    assert collector._calibration_buffer is None
    initial_sum = collector.hist.sum().item()
    assert initial_sum == 100

    third = torch.arange(100.0, 110.0).unsqueeze(0)
    collector.update(third)
    assert collector._calibration_buffer is None
    assert collector.hist.sum().item() == initial_sum + 10
