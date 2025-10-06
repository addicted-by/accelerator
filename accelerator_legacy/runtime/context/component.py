from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from omegaconf import DictConfig

from accelerator.utilities.base_container import BaseContainer
from accelerator.utilities.hydra_utils import instantiate
from accelerator.utilities.logging import get_logger
from accelerator.utilities.model_utils.unwrap import unwrap_model

T = TypeVar("T")
logger = get_logger(__name__)


class ComponentManager(BaseContainer):
    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})
        self._components: Dict[str, Any] = {}
        self._manual_components: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._dependency_map: Dict[str, List[str]] = {"model": ["optimizer", "scheduler"], "optimizer": ["scheduler"]}
        self._setup_default_factories()

    def _setup_default_factories(self):
        self._factories.update(
            {
                "model": self._create_model,
                "data": self._create_data,
                "optimizer": self._create_optimizer,
                "scheduler": self._create_scheduler,
                "callbacks": self._create_callbacks,
                "distributed": self._create_distributed,
            }
        )

    def register_factory(self, name: str, factory: Callable[[], Any]):
        self._factories[name] = factory

    def set_component(self, name: str, instance: Any):
        self._manual_components[name] = instance
        self._components[name] = self._process_component(name, instance)
        self._rebuild_dependents(name)

    def get_component(self, name: str) -> Optional[Any]:
        if name in self._components:
            return self._components[name]

        if name not in self._manual_components and name in self._factories:
            self._create_component_from_factory(name)

        return self._components.get(name)

    def has_component(self, name: str) -> bool:
        return name in self._components or name in self._manual_components

    def configure_dependencies(self, dependency_map: Dict[str, List[str]]):
        self._dependency_map = dependency_map

    def _process_component(self, name: str, instance: Any) -> Any:
        if name == "model":
            return self._process_model(instance)
        if name == "data":
            return self._process_data(instance)
        return instance

    def _process_data(self, data):
        from accelerator.runtime.datamodule.datamodule import DataModule

        if isinstance(data, DataModule):
            return data

        if isinstance(data, dict) and len(data.values()):
            return DataModule(**data)

        else:
            data_config = self.config.get("datamodule", {})
            return DataModule.initialize_from_config(data_config)

    def _process_model(self, model):
        from accelerator.runtime.model.accelerated_model import AcceleratedModel

        if isinstance(model, AcceleratedModel) or isinstance(unwrap_model(model), AcceleratedModel):
            return model
        else:
            model_config = self.config.get("model", {})
            return AcceleratedModel(model, model_config)

    def _create_component_from_factory(self, name: str):
        if name not in self._factories:
            return

        try:
            result = self._factories[name]()
            if result:
                self._components[name] = result
                logger.info(f"Created component: {name}")
        except Exception as e:
            logger.error(f"Failed to create component {name}: {e}")
            raise

    def _rebuild_dependents(self, changed_component: str):
        for dependent in self._dependency_map.get(changed_component, []):
            if dependent not in self._manual_components:
                self._create_component_from_factory(dependent)

    def _create_model(self):
        from accelerator.runtime.model.accelerated_model import AcceleratedModel

        if "model" not in self.config or self.config.model is None:
            return None
        core_model = instantiate(self.config.model.model_core)
        return AcceleratedModel(core_model, self.config.model)

    def _create_data(self):
        from accelerator.runtime.datamodule.datamodule import DataModule

        if "datamodule" not in self.config or self.config.datamodule is None:
            return None
        return DataModule.initialize_from_config(self.config.datamodule)

    def _create_optimizer(self):
        tc = self.config.get("training_components", {})
        if not tc or "optimizer" not in tc:
            return None

        model = self.get_component("model")
        if not model:
            return None

        return instantiate(tc.optimizer, params=[p for p in model.parameters() if p.requires_grad])

    def _create_scheduler(self):
        tc = self.config.get("training_components", {})
        if not tc or "scheduler" not in tc:
            return None

        optimizer = self.get_component("optimizer")
        if not optimizer:
            return None

        return instantiate(tc.scheduler, optimizer=optimizer)

    def _create_callbacks(self):
        from accelerator.runtime.callbacks.always_on import create_always_on_callbacks
        from accelerator.runtime.callbacks.manager import CallbackManager

        cb_cfg = self.config.get("callbacks", {})

        progress_bar = self.config.get("progress_bar")

        # Instantiate always-on callbacks first
        manager = CallbackManager(create_always_on_callbacks(progress_bar))

        # Append user defined callbacks, avoiding duplicates
        if cb_cfg:
            user_manager = CallbackManager.initialize_from_config(cb_cfg)
            for cb in user_manager.callbacks:
                if not any(isinstance(existing, cb.__class__) for existing in manager.callbacks):
                    manager.add_callback(cb)

        return manager

    def _create_distributed(self):
        if "engine" not in self.config:
            return None
        engine = instantiate(self.config.engine)
        return engine

    def clear(self):
        self._components.clear()
        self._manual_components.clear()

    def list_components(self) -> List[str]:
        return list(set(self._components.keys()) | set(self._manual_components.keys()))

    def _get_summary_info(self) -> str:
        total_factories = len(self._factories)
        active_components = len(self._components)
        return f"{active_components}/{total_factories} active"

    def _get_representation_sections(self) -> List[Tuple[str, List[str]]]:
        sections = []

        if self.config:
            config_sections = len(self.config) if hasattr(self.config, "__len__") else 1
            sections.append(("Configuration", [f"{config_sections} sections"]))

        if self._components:
            active_items = []
            for name, component in self._components.items():
                component_type = type(component).__name__
                source = "manual" if name in self._manual_components else "factory"
                active_items.append(f"{name}: {component_type} ({source})")
            sections.append(("Active Components", active_items))

        available_factories = set(self._factories.keys()) - set(self._components.keys())
        if available_factories:
            factory_items = list(sorted(available_factories))
            sections.append(("Available Factories", factory_items))

        if self._dependency_map:
            dependency_items = []
            for component, deps in self._dependency_map.items():
                deps_str = ", ".join(deps)
                dependency_items.append(f"{component} → {deps_str}")
            sections.append(("Dependencies", dependency_items))

        return sections
