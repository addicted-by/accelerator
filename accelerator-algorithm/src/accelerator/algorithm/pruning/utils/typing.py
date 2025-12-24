from pathlib import Path
from typing import AnyStr, Optional, TypeVar, Union

import torch

PathType = Optional[Union[AnyStr, Path]]
InOutType = Union[torch.Tensor, list[torch.Tensor]]
ModuleType = TypeVar("ModuleType", bound=torch.nn.Module)
