from accelerator.runtime.callbacks.manager import CallbackManager
from accelerator.runtime.checkpoint_manager import CheckpointManager
from accelerator.runtime.loop.base import LoopBase


class BaseTrainer:
    @staticmethod
    def train(context, config):
        callbacks: CallbackManager = context.callbacks
        
        with callbacks.phase('accelerate', context):
            context.model.accelerate(context)
        
        train_loop: LoopBase = context.training_manager.components.train_loop
        val_loop: LoopBase = context.training_manager.validation_components.val_loop
        test_loop: LoopBase = context.training_manager.test_components.test_loop
        
        with callbacks.phase('train', context):
            for i in range(config.total_epochs):
                overall_metrics = {}
                
                overall_metrics['train'] = train_loop.run_epoch(context)

                if i % config.validate_every == 0 and val_loop:
                    overall_metrics['val'] = val_loop.run_epoch(context)
                
                if i % config.test_every == 0 and test_loop:
                    overall_metrics['test'] = test_loop.run_epoch(context)

                if i % config.save_frequency == 0:
                    CheckpointManager.save_checkpoint(
                        model=context.model,
                        optimizer=context.optimizer,
                        save_dir=config.paths.checkpoint_dir,
                        metrics=overall_metrics,
                        cfg=config.checkpoint_save,
                        **context.training_manager.get_training_state_snapshot()
                    )