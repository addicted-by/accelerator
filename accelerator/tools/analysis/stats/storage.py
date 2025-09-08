from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch


def _tensor_to_list(tensor: torch.Tensor):
    return tensor.detach().cpu().tolist()


def save_tensor_stats(
    activations: Dict[str, Dict[str, torch.Tensor]],
    gradients: Dict[str, Dict[str, torch.Tensor]],
    path: str | Path,
) -> None:
    """Save collected tensor statistics to a JSON file.

    Parameters
    ----------
    activations: Mapping of node names to activation statistics tensors.
    gradients: Mapping of node names to gradient statistics tensors.
    path: Destination file path for serialized statistics.
    """
    serializable = {
        "activations": {
            node: {name: _tensor_to_list(t) for name, t in stats.items()} for node, stats in activations.items()
        },
        "gradients": {
            node: {name: _tensor_to_list(t) for name, t in stats.items()} for node, stats in gradients.items()
        },
    }
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(serializable, f, indent=2)
