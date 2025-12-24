from pathlib import Path
from typing import Any, AnyStr, Protocol, TypedDict, Union

import torch
from omegaconf import DictConfig

_DEVICE = Union[torch.device, str, int]

# Config type
ConfigType = Union[dict[str, Any], DictConfig]

# Path types
PathType = Union[AnyStr, Path]
InputPathType = PathType
OutputPathType = PathType

# Common tensor types
TensorType = Union[torch.Tensor, list[torch.Tensor]]
BatchTensorType = Union[torch.Tensor, tuple[torch.Tensor, ...], dict[str, torch.Tensor]]
ModelOutputType = Union[torch.Tensor, tuple[torch.Tensor, ...], dict[str, torch.Tensor]]
OptimizerStateType = dict[str, Any]
ModuleType = torch.nn.Module

# Phase metrics dict
PhaseMetricsDict = dict[str, Union[TensorType, float, list[float]]]


class MetricsDict(TypedDict):
    train: PhaseMetricsDict
    val: PhaseMetricsDict
    test: PhaseMetricsDict


class ModelProtocol(Protocol):
    def __call__(self, *args, **kwargs) -> Any:
        ...


class TrainingFunction(Protocol):
    def __call__(self, context) -> dict[str, float]:
        ...


class DistributedBackendProtocol(Protocol):
    @property
    def device(self) -> torch.device:
        ...

    def rank(self) -> int:
        ...

    def world_size(self) -> int:
        ...

    def is_main_process(self) -> bool:
        ...

    def barrier(self) -> None:
        ...

    def all_reduce(self, tensor: Any, op: str = "mean") -> Any:
        ...

    def gather(self, tensor: Any) -> Any:
        ...

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        ...
