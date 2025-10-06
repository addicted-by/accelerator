from pathlib import Path
from typing import List, Optional, TypeVar, Union, AnyStr
import torch


PathType = Optional[Union[AnyStr, Path]]
InOutType = Union[torch.Tensor, List[torch.Tensor]]
ModuleType = TypeVar("ModuleType", bound=torch.nn.Module)