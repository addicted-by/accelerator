from contextlib import contextmanager
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from accelerator.utilities.hydra_utils import instantiate
from accelerator.utilities.logging import get_logger

from .base import BaseCallback

if TYPE_CHECKING:
    from accelerator.core.context import Context


log = get_logger(__name__)


class CallbackManager:
    def __init__(self, callbacks: tuple[BaseCallback] = ()):
        """Initialize CallbackManager with a list of callbacks.

        Args:
            callbacks: List of BaseCallback instances
        """
        self.callbacks = sorted(callbacks, key=lambda x: x.priority)

    @staticmethod
    def initialize_from_config(callbacks_cfg: DictConfig) -> "CallbackManager":
        """Create CallbackManager from configuration.

        Args:
            callbacks_cfg: Configuration for callbacks

        Returns:
            CallbackManager instance
        """
        callback_instances = CallbackManager._instantiate_callbacks(callbacks_cfg)
        return CallbackManager(callback_instances)

    @staticmethod
    def _instantiate_callbacks(callbacks_cfg: DictConfig) -> list[BaseCallback]:
        """Instantiate callbacks from configuration.

        Args:
            callbacks_cfg: Configuration for callbacks

        Returns:
            List of instantiated callback objects
        """
        callbacks = []
        for callback_name in callbacks_cfg.active_callbacks:
            callback_cfg = callbacks_cfg[callback_name]
            callback_instance = instantiate(callback_cfg)
            if isinstance(callback_instance, BaseCallback):
                callbacks.append(callback_instance)
            else:
                log.warning(f"Instantiated callback {callback_name} is not a BaseCallback instance")
        return callbacks

    def _safe_execute(self, method_name: str, context):
        """Safely execute callback method with error handling."""
        for cb in self.callbacks:
            try:
                if hasattr(cb, method_name):
                    getattr(cb, method_name)(context)
            except Exception as e:
                log.error(f"Callback {cb.__class__.__name__} failed: {str(e)}")
                if cb.critical:
                    raise

    def trigger(self, event_name: str, context):
        """Trigger callbacks for a specific event.

        Args:
            event_name: Name of the event (without 'on_' prefix)
            context: Context object to pass to callbacks
        """
        self._safe_execute(f"on_{event_name}", context)

    def add_callback(self, callback: BaseCallback):
        """Add a callback to the manager.

        Args:
            callback: BaseCallback instance to add
        """
        if not isinstance(callback, BaseCallback):
            raise ValueError("callback must be a BaseCallback instance")

        self.callbacks.append(callback)
        self.callbacks.sort(key=lambda x: x.priority)

    def remove_callback(self, callback_class: type) -> bool:
        """Remove callback by class type.

        Args:
            callback_class: Class type of callback to remove

        Returns:
            True if callback was found and removed, False otherwise
        """
        for i, cb in enumerate(self.callbacks):
            if isinstance(cb, callback_class):
                self.callbacks.pop(i)
                return True
        return False

    @contextmanager
    def phase(self, name: str, context: "Context"):
        self._current_phase = name
        hook_name_begin = f"{name}_begin"
        hook_name_end = f"{name}_end"

        self.trigger(hook_name_begin, context)
        try:
            yield
        finally:
            self.trigger(hook_name_end, context)
            self._current_phase = None

    @property
    def callback_count(self) -> int:
        """Get number of registered callbacks."""
        return len(self.callbacks)

    def __repr__(self) -> str:
        callback_names = [cb.__class__.__name__ for cb in self.callbacks]
        return f"CallbackManager({self.callback_count} callbacks: {callback_names})"
