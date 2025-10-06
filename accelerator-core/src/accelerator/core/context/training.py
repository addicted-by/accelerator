from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from accelerator.core.distillation.manager import DistillationManager

from accelerator.utilities.logging import get_logger
from accelerator.utilities.base_container import BaseContainer, StateContainer, ComponentContainer
from accelerator.utilities.typings import PhaseMetricsDict


from .component import ComponentManager


logger = get_logger(__name__)


@dataclass
class TrainingComponents(ComponentContainer):
    optimizer: Any = None
    scheduler: Any = None
    loss_combiner: Any = None
    distillation_manager: Optional[DistillationManager] = None
    train_loop: Any = None

    def _get_component_mapping(self) -> Dict[str, Any]:
        return {
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'loss_combiner': self.loss_combiner,
            'distillation_manager': self.distillation_manager,
            'train_loop': self.train_loop
        }


@dataclass
class ValidationComponents(ComponentContainer):
    val_loop: Any = None
    config: Any = None

    def _get_component_mapping(self) -> Dict[str, Any]:
        return {
            'val_loop': self.val_loop,
            'config': self.config
        }


@dataclass
class TestComponents(ComponentContainer):
    test_loop: Any = None
    config: Any = None


    def _get_component_mapping(self) -> Dict[str, Any]:
        return {
            'test_loop': self.test_loop,
            'config': self.config
        }


@dataclass  
class TrainingState(StateContainer):
    current_epoch: int = 0
    current_step: int = 0
    global_epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None

    total_epochs: int = 0
    
    optimizer_state: Optional[dict] = None
    scheduler_state: Optional[dict] = None
    
    training_metrics: PhaseMetricsDict = field(default_factory=dict)
    validation_metrics: PhaseMetricsDict = field(default_factory=dict)
    test_metrics: PhaseMetricsDict = field(default_factory=dict)
    
    @property
    def snapshot(self):
        return asdict(self)

    def advance_step(self):
        self.current_step += 1
        self.global_step += 1
    
    def advance_epoch(self):
        self.current_epoch += 1
        self.global_epoch += 1
        self.current_step = 0

    def _get_progression_info(self) -> Dict[str, Any]:
        progression = {
            'epoch': self.current_epoch,
            'step': self.current_step
        }
        
        if self.best_metric is not None:
            progression['best_metric'] = self.best_metric
        
        return progression
    
    def _get_state_metrics(self) -> Dict[str, Any]:
        metrics = {}
        
        if self.training_metrics:
            metrics['Training Metrics'] = self.training_metrics
        
        if self.validation_metrics:
            metrics['Validation Metrics'] = self.validation_metrics
        
        if self.learning_rates:
            metrics['Learning Rates'] = self.learning_rates
        
        return metrics
    
    def _get_additional_sections(self) -> List[Tuple[str, List[str]]]:
        sections = []
        
        state_indicators = []
        if self.optimizer_state is not None:
            state_indicators.append("optimizer")
        if self.scheduler_state is not None:
            state_indicators.append("scheduler")
        
        if state_indicators:
            sections.append(("Saved States", state_indicators))
        
        return sections

class TrainingManager(BaseContainer):
    """Class used for Managing Training State"""
    def __init__(self):
        self.components = TrainingComponents()
        self.validation_components = ValidationComponents()
        self.test_components = TestComponents()

        self.state = TrainingState()
        self._component_manager = None
        
    def set_component_manager(self, manager: ComponentManager):
        self._component_manager = manager
    
    def set_optimizer(self, optimizer: Any) -> None:
        self.components.optimizer = optimizer

        if self._component_manager:
            self._component_manager.set_component('optimizer', optimizer)
    
    def set_scheduler(self, scheduler: Any) -> None:
        self.components.scheduler = scheduler

        if self._component_manager:
            self._component_manager.set_component('scheduler', scheduler)
    
    def set_loss_combiner(self, loss_combiner: Any) -> None:
        self.components.loss_combiner = loss_combiner
    
    def set_distillation_manager(self, distillation_manager: Any) -> None:
        self.components.distillation_manager = distillation_manager
    
    def get_optimizer(self) -> Any:
        if self.components.optimizer:
            return self.components.optimizer
        
        if self._component_manager:
            optimizer = self._component_manager.get_component('optimizer')
            if optimizer:
                self.components.optimizer = optimizer
            return optimizer
        return None
    
    def get_scheduler(self) -> Any:
        if self.components.scheduler:
            return self.components.scheduler
        
        if self._component_manager:
            scheduler = self._component_manager.get_component('scheduler')
            if scheduler:
                self.components.scheduler = scheduler
            return scheduler
        return None
    
    def advance_training_step(self) -> None:
        self.state.advance_step()
        logger.debug(f"Advanced to step {self.state.current_step}")
    
    def advance_training_epoch(self) -> None:
        self.state.advance_epoch()
        logger.info(f"Advanced to epoch {self.state.current_epoch}")
    
    def update_metrics(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key == 'training_metrics':
                self.state.training_metrics.update(value)
            elif key == 'validation_metrics':
                self.state.validation_metrics.update(value)
            elif hasattr(self.state, key):
                setattr(self.state, key, value)
    
    def get_training_state_snapshot(self) -> Dict[str, Any]:
        return self.state.snapshot
    
    def restore_training_state(self, state_snapshot: Dict[str, Any]) -> None:
        for key, value in state_snapshot.items():
            if hasattr(self.state, key):
                if isinstance(value, dict):
                    getattr(self.state, key).clear()
                    getattr(self.state, key).update(value)
                else:
                    setattr(self.state, key, value)
        
        if 'optimizer_state' in state_snapshot and self.components.optimizer:
            self.components.optimizer.load_state_dict(state_snapshot['optimizer_state'])
            
        if 'scheduler_state' in state_snapshot and self.components.scheduler:
            self.components.scheduler.load_state_dict(state_snapshot['scheduler_state'])
      
    def reset_state(self) -> None:
        self.state = TrainingState()
        self.components = TrainingComponents()
        logger.info("Training manager state reset")
    
    @property
    def current_epoch(self) -> int:
        return self.state.current_epoch
    
    @property
    def current_step(self) -> int:
        return self.state.current_step
    
    @property
    def training_metrics(self) -> Dict[str, Any]:
        return self.state.training_metrics
    
    @property
    def validation_metrics(self) -> Dict[str, Any]:
        return self.state.validation_metrics
    

    def _get_summary_info(self) -> str:
        component_manager_status = "integrated" if self._component_manager else "standalone"
        epoch_info = f"epoch={self.state.current_epoch}"
        step_info = f"step={self.state.current_step}"
        return f"{component_manager_status}, {epoch_info}, {step_info}"
    
    def _get_representation_sections(self) -> List[Tuple[str, List[str]]]:
        sections = []
        
        if self._component_manager:
            manager_repr = str(self._component_manager).replace('\n', '\n  ')
            sections.append(("Component Manager Integration", [manager_repr]))
        
        training_repr = str(self.components).replace('\n', '\n  ')
        sections.append(("Training Components", [training_repr]))
        
        if any([self.validation_components.val_loop, self.validation_components.config]):
            validation_repr = str(self.validation_components).replace('\n', '\n  ')
            sections.append(("Validation Components", [validation_repr]))
        
        if any([self.test_components.test_loop, self.test_components.config]):
            test_repr = str(self.test_components).replace('\n', '\n  ')
            sections.append(("Test Components", [test_repr]))
        
        state_repr = str(self.state).replace('\n', '\n  ')
        sections.append(("Training State", [state_repr]))
        
        capabilities = []
        if self.components.optimizer:
            capabilities.append("optimization")
        if self.components.scheduler:
            capabilities.append("scheduling")
        if self.components.loss_combiner:
            capabilities.append("loss_combination")
        if self.components.distillation_manager:
            capabilities.append("distillation")
        
        if capabilities:
            sections.append(("Active Capabilities", capabilities))
        
        return sections