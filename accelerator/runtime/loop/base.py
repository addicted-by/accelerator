from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING
from accelerator.typings.base import MetricsDict, PhaseMetricsDict

if TYPE_CHECKING:
    from accelerator.runtime.context.context import Context


class LoopBase(ABC):
    def __init__(self):
        self._current_phase = 'train'
    
    @abstractmethod
    def run_epoch(self, context: 'Context') -> MetricsDict:
        pass
    
    @abstractmethod
    def process_batch(self, batch: Any, context: 'Context') -> PhaseMetricsDict:
        pass
    
    def _update_metrics(self, metrics: Dict[str, float], context: 'Context'):
        context.update_training_state(metrics=metrics)