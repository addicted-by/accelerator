import abc
from textwrap import dedent
from typing import Any, Dict, Optional, TYPE_CHECKING
from accelerator.utilities.api_desc import APIDesc
from accelerator.utilities.hashable import _HashableConfigMixin

if TYPE_CHECKING:
    from accelerator.runtime.context import Context


@APIDesc.developer(dev_info="Ryabykin Alexey r00926208")
@APIDesc.status(status_level="Internal use only")
class AccelerationOperationBase(abc.ABC, _HashableConfigMixin):
    """Base class for acceleration operations."""

    def __init__(self, acceleration_config: Optional[Dict[str, Any]] = None):
        self.config = acceleration_config or {}
        self._meta_data = {}
        self.registry_type = None
        self._name = self.__class__.__name__

        self._not_loaded_error_msg: str = dedent(
            f"""Please, load state dict before (re)-applying
            the acceleration {self._name}
            """
        )

    @abc.abstractmethod
    def apply(self, context: "Context") -> None:
        """Apply the acceleration operation to the model."""

    @abc.abstractmethod
    def reapply(self, model) -> None:
        """Reapply the acceleration operation to the model."""

    @abc.abstractmethod
    def calibrate(self, context: "Context") -> None:
        """Apply calibrates logic to the model."""

    def state_dict(self) -> Dict[str, Any]:
        """Return the state dictionary of the acceleration operation."""
        return {"config": self.config, "__meta_data__": self._meta_data}

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load the state dictionary of the acceleration operation."""
        self.config = state.get("config", self.config)
        self._meta_data = state.get("__meta_data__", self._meta_data)
