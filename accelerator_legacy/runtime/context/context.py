from typing import List, Optional, Tuple
from omegaconf import DictConfig
from accelerator.runtime.callbacks.manager import CallbackManager
from accelerator.runtime.checkpoint import CheckpointManager
from accelerator.runtime.datamodule.datamodule import DataModule
from accelerator.runtime.model.accelerated_model import AcceleratedModel
from accelerator.utilities.rich_utils.config_tree import print_config_tree
from accelerator.utilities.distributed_state.state import distributed_state
from accelerator.utilities.logging import get_logger
from accelerator.utilities.base_container import BaseContainer
from accelerator.runtime.engine.base import DistributedBackend

from .component import ComponentManager
from .training import TrainingManager


logger = get_logger(__name__)


class Context(BaseContainer):
    """Class used only as a container for all required components"""
    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})
        self.components = ComponentManager(self.config)

        self.training_manager = TrainingManager()
        self.training_manager.set_component_manager(self.components)

        self.is_distributed = False
        self.distributed_engine = None

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
        self.distributed_engine: DistributedBackend = self.components.get_component('distributed')
        if self.distributed_engine:
            distributed_state.set_engine(self.distributed_engine)
            self.distributed_engine.setup()
            self.is_distributed = True

    def initialize(self, checkpoint_path: Optional[str] = None) -> None:
        self.initialize_single(checkpoint_path)
        
    def make_distributed(self):
        if self.is_distributed and self.model:
            self.model = self.distributed_engine.prepare_model(self.model)
            self.components.set_component('model', self.model)
        
        if self.is_distributed and self.data:
            for loader_name in self.data.loader_names:
                self.data._dataloaders[loader_name] = self.distributed_engine.prepare_dataloader(self.data._dataloaders[loader_name])
        
        if self.is_distributed and self.optimizer:
            self.optimizer = self.distributed_engine.prepare_optimizer(self.optimizer)
            self.training_components.optimizer = self.optimizer
        
    def set_model(self, model) -> 'Context':
        self.components.set_component('model', model)
        return self

    def set_optimizer(self, optimizer) -> 'Context':
        self.components.set_component('optimizer', optimizer)
        self.training_manager.set_optimizer(optimizer)
        return self

    def set_scheduler(self, scheduler) -> 'Context':
        self.components.set_component('scheduler', scheduler)
        self.training_manager.set_scheduler(scheduler)
        self.callbacks.add_callback(scheduler)
        return self

    def set_data(self, data) -> 'Context':
        self.components.set_component('data', data)
        return self

    def set_callbacks(self, callbacks) -> 'Context':
        self.components.set_component('callbacks', callbacks)
        return self

    # def set_checkpoint_manager(self, checkpoint_manager) -> 'Context':
    #     self.components.set_component('checkpoint', checkpoint_manager)
    #     return self

    @property
    def model(self) -> AcceleratedModel:
        return self.components.get_component('model')

    @property
    def data(self) -> DataModule:
        return self.components.get_component('data')

    @property
    def optimizer(self):
        optimizer = self.components.get_component('optimizer')
        
        if optimizer:
            self.training_manager.set_optimizer(optimizer)

        return optimizer

    @property
    def scheduler(self):
        scheduler = self.components.get_component('scheduler')
        if scheduler:
            self.training_manager.set_scheduler(scheduler)
            self.callbacks.add_callback(scheduler)
        return scheduler

    @property
    def callbacks(self) -> CallbackManager:
        return self.components.get_component('callbacks')

    # @property
    # def checkpoint_manager(self) -> CheckpointManager:
    #     return self.components.get_component('checkpoint')
    @property
    def checkpoint_manager(self):
        return CheckpointManager
    

    def initialize_single(self, checkpoint_path: Optional[str] = None) -> None:
        self._print_config()

        self.model
        self.data
        self.callbacks
        self.optimizer
        self.scheduler

        if checkpoint_path and self.model:
            result = self.checkpoint_manager.load_checkpoint(
                path=checkpoint_path, 
                model=self.model,
                cfg_override=self.config.get('checkpoint_load', None)
            )
            if 'training_state' in result:
                self.training_manager.restore_training_state(result['training_state'])

            return result

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
    
    def _get_representation_sections(self) -> List[Tuple[str, List[str]]]:
        sections = []
        
        if self.config:
            config_info = f"{len(self.config)} sections" if hasattr(self.config, '__len__') else "loaded"
            sections.append(("Configuration", [config_info]))
        
        if self.is_distributed:
            distributed_info = []
            distributed_info.append(f"engine: {type(self.distributed_engine).__name__}")
            distributed_info.append(f"rank: {self.distributed_engine.rank() if hasattr(self.distributed_engine, 'rank') else 'unknown'}")
            distributed_info.append(f"world_size: {self.distributed_engine.world_size() if hasattr(self.distributed_engine, 'world_size') else 'unknown'}")
            sections.append(("Distributed Training", distributed_info))
        
        component_repr = str(self.components).replace('\n', '\n  ')
        sections.append(("Component Management", [component_repr]))
        
        training_repr = str(self.training_manager).replace('\n', '\n  ')
        sections.append(("Training Management", [training_repr]))
        
        return sections