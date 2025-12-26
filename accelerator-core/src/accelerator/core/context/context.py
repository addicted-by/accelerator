from typing import Any, Optional

from omegaconf import DictConfig

from accelerator.core.callbacks import CallbackManager
from accelerator.core.checkpoint import CheckpointManager
from accelerator.core.datamodule import DataModule
from accelerator.core.engine import DistributedBackend
from accelerator.core.model import AcceleratedModel
from accelerator.utilities.base_container import BaseContainer
from accelerator.utilities.distributed_state import distributed_state
from accelerator.utilities.logging import get_logger
from accelerator.utilities.rich_utils.config_tree import print_config_tree

from .component import ComponentManager
from .containers import (
    PathResolutionError,
    PerBatchContainer,
    PerEpochContainer,
    PersistentContainer,
    PerStepContainer,
)
from .training import TrainingManager

logger = get_logger(__name__)


class Context(BaseContainer):
    """Class used only as a container for all required components."""

    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})

        self.per_batch = PerBatchContainer()
        self.per_step = PerStepContainer()
        self.per_epoch = PerEpochContainer()
        self.persistent = PersistentContainer()

        self._CONTAINERS = {
            "per_batch": self.per_batch,
            "per_step": self.per_step,
            "per_epoch": self.per_epoch,
            "persistent": self.persistent,
        }

        self.components = ComponentManager(self.config)

        self.training_manager = TrainingManager()
        self.training_manager.set_component_manager(self.components)

        self.is_distributed = False
        self.distributed_engine = None

    def get_item(self, path: str) -> Any:
        """Get item using hierarchical path like 'per_step.input.rgb'.

        Args:
            path: Hierarchical path in format 'lifecycle_scope.sub_container.key'

        Returns:
            The value at the specified path

        Raises:
            ValueError: If path doesn't include lifecycle scope
            PathResolutionError: If the path cannot be resolved

        Examples:
            >>> context.get_item('per_batch.input.rgb')
            >>> context.get_item('per_step.loss.total')
            >>> context.get_item('persistent.model.state')

        """
        parts = path.split(".", 1)
        if len(parts) < 2:
            raise ValueError(f"Path must include lifecycle scope: {path}")

        scope, sub_path = parts
        container = self._get_lifecycle_container(scope)
        return container.get_item(sub_path)

    def set_item(self, path: str, value: Any, use_weakref: Optional[bool] = None) -> None:
        """Set item using hierarchical path.

        Args:
            path: Hierarchical path in format 'lifecycle_scope.sub_container.key'
            value: The value to set
            use_weakref: If True, force weak ref; if False, force strong ref;
                        if None, use automatic detection

        Raises:
            ValueError: If path doesn't include lifecycle scope
            PathResolutionError: If the path cannot be resolved

        Examples:
            >>> context.set_item('per_batch.input.rgb', tensor)
            >>> context.set_item('per_step.loss.total', 0.5, use_weakref=False)
            >>> context.set_item('persistent.additional.pca_mat', matrix)

        """
        parts = path.split(".", 1)
        if len(parts) < 2:
            raise ValueError(f"Path must include lifecycle scope: {path}")

        scope, sub_path = parts
        container = self._get_lifecycle_container(scope)
        container.set_item(sub_path, value, use_weakref)

    def _get_lifecycle_container(self, scope: str):
        """Get lifecycle container by scope name.

        Args:
            scope: The lifecycle scope name ('per_batch', 'per_step', 'per_epoch', 'persistent')

        Returns:
            The corresponding lifecycle container

        Raises:
            ValueError: If scope is not recognized

        """
        if scope not in self._CONTAINERS:
            raise ValueError(f"Unknown lifecycle scope: {scope}. " f"Available: {list(self._CONTAINERS.keys())}")
        return self._CONTAINERS[scope]

    def update_container(self, scope: str, update_dict: dict):
        if scope not in self._CONTAINERS:
            raise ValueError(f"Unknown lifecycle scope: {scope}. " f"Available: {list(self._CONTAINERS.keys())}")

        self._CONTAINERS[scope].update(update_dict)

    @property
    def batch_count(self):
        return len(self.data.train_loader)

    @property
    def total_epochs(self):
        return self.training_manager.state.total_epochs

    def set_total_epochs(self, total_epochs):
        self.training_manager.state.total_epochs = total_epochs

    @distributed_state.on_main_process
    def _print_config(self) -> None:
        if self.config:
            print_config_tree(self.config)

    def setup_engine(self):
        self.distributed_engine: DistributedBackend = self.components.get_component("distributed")
        if self.distributed_engine:
            distributed_state.set_engine(self.distributed_engine)
            self.distributed_engine.setup()
            self.is_distributed = True

    def initialize(self, checkpoint_path: Optional[str] = None) -> None:
        self.initialize_single(checkpoint_path)

    def make_distributed(self):
        if self.is_distributed and self.model:
            self.model = self.distributed_engine.prepare_model(self.model)
            self.components.set_component("model", self.model)

        if self.is_distributed and self.data:
            for loader_name in self.data.loader_names:
                self.data._dataloaders[loader_name] = self.distributed_engine.prepare_dataloader(
                    self.data._dataloaders[loader_name]
                )

        if self.is_distributed and self.optimizer:
            self.optimizer = self.distributed_engine.prepare_optimizer(self.optimizer)
            self.training_components.optimizer = self.optimizer

    def set_model(self, model) -> "Context":
        self.components.set_component("model", model)
        return self

    def set_optimizer(self, optimizer) -> "Context":
        self.components.set_component("optimizer", optimizer)
        self.training_manager.set_optimizer(optimizer)
        return self

    def set_scheduler(self, scheduler) -> "Context":
        self.components.set_component("scheduler", scheduler)
        self.training_manager.set_scheduler(scheduler)
        self.callbacks.add_callback(scheduler)
        return self

    def set_data(self, data) -> "Context":
        self.components.set_component("data", data)
        return self

    def set_callbacks(self, callbacks) -> "Context":
        self.components.set_component("callbacks", callbacks)
        return self

    # def set_checkpoint_manager(self, checkpoint_manager) -> 'Context':
    #     self.components.set_component('checkpoint', checkpoint_manager)
    #     return self

    @property
    def model(self) -> AcceleratedModel:
        """Backward compatible access to model.

        Tries to get model from persistent container first, falls back to
        component manager for backward compatibility.
        """
        try:
            return self.get_item("persistent.model.instance")
        except (ValueError, PathResolutionError, KeyError):
            return self.components.get_component("model")

    @model.setter
    def model(self, value):
        """Backward compatible model setter.

        Sets model in both persistent container and component manager
        for backward compatibility.
        """
        self.set_item("persistent.model.instance", value)
        self.components.set_component("model", value)

    @property
    def data(self) -> DataModule:
        return self.components.get_component("data")

    @property
    def optimizer(self):
        """Backward compatible access to optimizer.

        Tries to get optimizer from persistent container first, falls back to
        component manager for backward compatibility.
        """
        try:
            optimizer = self.get_item("persistent.optimizer.instance")
        except (ValueError, PathResolutionError, KeyError):
            optimizer = self.components.get_component("optimizer")

        if optimizer:
            self.training_manager.set_optimizer(optimizer)

        return optimizer

    @optimizer.setter
    def optimizer(self, value):
        """Backward compatible optimizer setter."""
        self.set_item("persistent.optimizer.instance", value)
        self.components.set_component("optimizer", value)
        self.training_manager.set_optimizer(value)

    @property
    def scheduler(self):
        """Backward compatible access to scheduler.

        Tries to get scheduler from persistent container first, falls back to
        component manager for backward compatibility.
        """
        try:
            scheduler = self.get_item("persistent.scheduler.instance")
        except (ValueError, PathResolutionError, KeyError):
            scheduler = self.components.get_component("scheduler")

        if scheduler:
            self.training_manager.set_scheduler(scheduler)
            self.callbacks.add_callback(scheduler)
        return scheduler

    @scheduler.setter
    def scheduler(self, value):
        """Backward compatible scheduler setter."""
        self.set_item("persistent.scheduler.instance", value)
        self.components.set_component("scheduler", value)
        self.training_manager.set_scheduler(value)
        self.callbacks.add_callback(value)

    @property
    def callbacks(self) -> CallbackManager:
        return self.components.get_component("callbacks")

    # @property
    # def checkpoint_manager(self) -> CheckpointManager:
    #     return self.components.get_component('checkpoint')
    @property
    def checkpoint_manager(self):
        return CheckpointManager

    def initialize_single(self, checkpoint_path: Optional[str] = None) -> None:
        self._print_config()

        _ = self.model
        _ = self.data
        _ = self.callbacks
        _ = self.optimizer
        _ = self.scheduler

        if checkpoint_path and self.model:
            result = self.checkpoint_manager.load_checkpoint(
                path=checkpoint_path,
                model=self.model,
                cfg_override=self.config.get("checkpoint_load", None),
            )
            if "training_state" in result:
                self.training_manager.restore_training_state(result["training_state"])

            return result

    def on_batch_start(self) -> None:
        """Called at the start of each batch.

        Clears per_batch container to prepare for new batch data.
        """
        self.per_batch.clear()

    def on_batch_end(self) -> None:
        """Called at the end of each batch.

        Cleans up dead weak references in per_batch container.
        """
        self.per_batch.clear()
        self.per_batch.cleanup_dead_refs()

    def on_step_end(self) -> None:
        """Called after optimizer.step().

        Clears per_step container and cleans up dead weak references.
        """
        self.per_step.clear()
        self.per_step.cleanup_dead_refs()

    def on_epoch_end(self) -> None:
        """Called at the end of each epoch.

        Clears per_epoch container and cleans up dead weak references.
        """
        self.per_epoch.clear()
        self.per_epoch.cleanup_dead_refs()

    def cleanup(self):
        if self.distributed_engine:
            self.distributed_engine.cleanup()

        self.components.clear()
        self.training_manager.reset_state()

    def _get_summary_info(self) -> str:
        distributed_status = "distributed" if self.is_distributed else "single-node"
        engine_type = type(self.distributed_engine).__name__ if self.distributed_engine else "None"
        training_info = f"epoch={self.training_manager.current_epoch}"
        return f"{distributed_status}, engine={engine_type}, {training_info}"

    def _get_representation_sections(self) -> list[tuple[str, list[str]]]:
        sections = []

        if self.config:
            config_info = f"{len(self.config)} sections" if hasattr(self.config, "__len__") else "loaded"
            sections.append(("Configuration", [config_info]))

        if self.is_distributed:
            distributed_info = []
            distributed_info.append(f"engine: {type(self.distributed_engine).__name__}")
            distributed_info.append(
                f"rank: {self.distributed_engine.rank() if hasattr(self.distributed_engine, 'rank') else 'unknown'}"
            )
            distributed_info.append(
                f"world_size: {self.distributed_engine.world_size() if hasattr(self.distributed_engine, 'world_size') else 'unknown'}"
            )
            sections.append(("Distributed Training", distributed_info))

        component_repr = str(self.components).replace("\n", "\n  ")
        sections.append(("Component Management", [component_repr]))

        training_repr = str(self.training_manager).replace("\n", "\n  ")
        sections.append(("Training Management", [training_repr]))

        return sections
