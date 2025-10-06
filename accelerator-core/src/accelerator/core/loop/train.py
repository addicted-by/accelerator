from typing import TYPE_CHECKING


from .base import LoopBase


from accelerator.core.callbacks import CallbackManager
from accelerator.utilities.distributed_state import distributed_state

from accelerator.utilities.typings import MetricsDict, PhaseMetricsDict
from accelerator.utilities.move_to_device import move_data_to_device

if TYPE_CHECKING:
    from accelerator.core.context import Context


class TrainLoop(LoopBase):
    def __init__(self):
        super().__init__()
    
    def run_epoch(self, context: 'Context') -> MetricsDict:
        callbacks: CallbackManager = context.callbacks

        with callbacks.phase('train_epoch', context):
            dataloader = context.data.train_loader
            
            for _, batch in enumerate(dataloader):
                batch_metrics = self.process_batch(batch, context)
                self._update_metrics(batch_metrics, context)
                        
            return {
                'accumulated_loss': context.loss_combiner.accumulated_loss,
                'last_loss': context.loss_combiner.last_loss
            }
    
    def process_batch(self, batch, context: 'Context') -> PhaseMetricsDict:
        callbacks: CallbackManager = context.callbacks

        with callbacks.phase('train_batch', context):
            context.model.train()
            context.optimizer.zero_grad()
            
            inputs, targets, additional = batch

            inputs = move_data_to_device(inputs, distributed_state.device)
            targets = move_data_to_device(targets, distributed_state.device)
            additional = move_data_to_device(additional, distributed_state.device)

            predictions = context.model(*inputs)

            if context.distillation_manager:
                targets.update(context.distillation_manager(*inputs, **additional))
            
            loss = context.loss_combiner.add_get_loss(
                predictions, targets,
                inputs=inputs,
                additional=additional,
                context=context
            )
            
            reg_term = getattr(context.model, 'regularization_term', None)
            if reg_term is None:
                ...
            else:
                loss += reg_term
            
            with callbacks.phase('backward', context):
                loss.backward()
            
            with callbacks.phase('optimizer_step', context):
                context.optimizer.step()
            
            return {
                'loss': loss.item(),
                'batch_size': inputs.shape[0]
            }