from flatten_dict import flatten
from accelerator.runtime.pipeline.step import (
    StepConfigManager, make_step_cfg,
    resolve_checkpoint_path
)
import hydra
from omegaconf import OmegaConf, DictConfig
import mlflow
from typing import Tuple
from accelerator.utilities.experimental import get_experiment_tags
from accelerator.utilities.seed import setup_seed
from accelerator.utilities.logging import get_logger
import os



log = get_logger(__name__)

def get_distributed_args(cfg: DictConfig) -> Tuple[int, int]:
    """
    PLACEHOLDER. NEED TO BE IMPROVED FOR FURTHER!
    """
    nnodes = int(getattr(cfg, "nnodes", 1))
    nproc  = int(getattr(cfg, "nproc_per_node", 8))
    return nnodes, nproc


@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    log.info("\n=== FULL EXPERIMENT CONFIG ===")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    experiment_id = os.getenv('EXPERIMENT_ID', None)


    cm = None
    for step_name in cfg.pipeline.active_steps:
        run_cfg = make_step_cfg(cfg, step_name)
        run_name = run_cfg.job_name + "_" + run_cfg.step_name
        parent_id = run_cfg.parent_id
        step_id = run_cfg.attempt_id

        resolve_checkpoint_path(run_cfg, cm)

        cm = StepConfigManager.from_cfg(run_cfg)

        log.info(f"\n=== CONFIG FOR {step_name} ===")
        log.info(f">>> TRAINING {run_name} with config {cm.config_path}")

        nnodes, nproc = get_distributed_args(run_cfg)
        
        tags_nested_run = get_experiment_tags({
            'mlflow.run_id': step_id,
            'mlflow.artifact_location': run_cfg.paths.experiment_dir,
        })

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags_nested_run,
            parent_run_id=parent_id,
            nested=True,
        ):
            mlflow.log_params(flatten(run_cfg, reducer='dot', max_flatten_depth=4), synchronous=True)

            mlflow.projects.run(
                uri=".",
                entry_point=cfg.entry_point,
                parameters={
                    "cfg": str(cm.config_path),
                    "nnodes": str(nnodes),
                    "nproc_per_node": str(nproc),
                },
                experiment_id=experiment_id,
                run_id=step_id,
                synchronous=True,
                env_manager="local",
            )


if __name__ == "__main__":
    setup_seed(1337)
    main()