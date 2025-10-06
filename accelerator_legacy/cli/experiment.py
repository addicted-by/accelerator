import omegaconf
from typing import Dict, Any
from accelerator.runtime.context import Context
from accelerator.utilities.distributed_state import distributed_state
from accelerator.utilities.seed import setup_seed
from accelerator.utilities.hydra_utils import instantiate


_default_trainer_setup = dict(
    _target_='accelerator.tools.finetuning.BaseTrainer'
)


def setup_context(config_path):
    config = omegaconf.OmegaConf.load(config_path)
    if config.seed is not None:
        setup_seed(config.seed)

    context = Context(config)
    context.setup_engine()
    
    context.initialize(config.paths.checkpoint_path) # change CheckpointManager loading    
    return context, config


class Experiment:
    def train(self, config_path, trainer_setup: Dict[str, Any] = _default_trainer_setup):
        context, config = setup_context(config_path)
        trainer = instantiate(trainer_setup)

        print(distributed_state.device)
        trainer.train(context, config)

    def analyze_model(self, config_path):
        context, config = setup_context(config_path)

    def analyze_data(self, config_path):
        context, config = setup_context(config_path)