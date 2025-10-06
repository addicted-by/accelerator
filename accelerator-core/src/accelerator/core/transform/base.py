import abc
from typing import Any, Dict, Optional, Tuple

from accelerator.utilities.distributed_state import distributed_state

from accelerator.utilities.api_desc import APIDesc
from accelerator.utilities.hashable import _HashableConfigMixin


@APIDesc.developer(dev_info='Ryabykin Alexey r00926208')
@APIDesc.status(status_level='Internal use only')
class BaseTransform(abc.ABC, _HashableConfigMixin):
    """
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        """
        self.config: Dict[str, Any] = config or {}
        self.name: str = self.config.get('name', self.__class__.__name__)
    
    @abc.abstractmethod
    def apply(self, *args, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        """
        pass

    @property
    def device(self):
        return distributed_state.device

    def __call__(self, *args, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        return self.apply(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"