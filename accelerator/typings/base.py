from pathlib import Path
from omegaconf import DictConfig
import torch
from typing import Any, AnyStr, Dict, List, TypeVar, Protocol, Tuple, TypedDict, Union, TYPE_CHECKING


if TYPE_CHECKING:
    from accelerator.runtime.context import Context as _Context
else:
    class _Context:  # type: ignore[revived-private-name]
        ...

CONTEXT = TypeVar('CONTEXT', bound=_Context)
_DEVICE = Union[torch.device, str, int]


# Config type
ConfigType = Union[Dict[str, Any], DictConfig]

# Path types
PathType = Union[AnyStr, Path]
InputPathType = PathType
OutputPathType = PathType

# Common tensor types
TensorType = Union[torch.Tensor, List[torch.Tensor]]
BatchTensorType = Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]
ModelOutputType = Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]
OptimizerStateType = Dict[str, Any]
ModuleType = torch.nn.Module

# Phase metrics dict
PhaseMetricsDict = Dict[str, Union[TensorType, float, List[float]]]

class MetricsDict(TypedDict):
    train: PhaseMetricsDict
    val: PhaseMetricsDict
    test: PhaseMetricsDict

class ModelProtocol(Protocol):
    def __call__(self, *args, **kwargs) -> Any: ...

class TrainingFunction(Protocol):
    def __call__(self, context) -> Dict[str, float]: ...