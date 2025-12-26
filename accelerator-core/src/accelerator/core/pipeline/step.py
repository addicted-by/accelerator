from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf, open_dict

from accelerator.utilities.hydra_utils import _compose_with_existing_hydra
from accelerator.utilities.logging import get_logger
from accelerator.utilities.rich_utils.config_tree import print_config_tree

log = get_logger(__name__)


def resolve_checkpoint_path(
    run_cfg: DictConfig,
    prev_cm: Optional["StepConfigManager"],
) -> None:
    """If `run_cfg.load_from_previous` is True and a previous StepConfigManager
    is available, copy the checkpoint directory into
    `run_cfg.paths.checkpoint_path`.

    This mutates *run_cfg* in-place inside an `open_dict` context so we
    respect OmegaConf's struct mode.
    """
    if not getattr(run_cfg, "load_from_previous", False):
        return

    if prev_cm is None:
        log.warning("load_from_previous=True but no previous step exists; " "keeping the checkpoint_path unchanged.")
        return

    with open_dict(run_cfg.paths):
        run_cfg.paths.checkpoint_path = prev_cm.ckpt_dir


def make_step_cfg(full_cfg: DictConfig, step_name: str) -> DictConfig:
    """PLACEHOLDER."""
    meta = full_cfg.pipeline.active_steps[step_name]

    # step_cfg = copy.deepcopy(full_cfg)
    # OmegaConf.set_struct(step_cfg, False)

    # step_cfg = OmegaConf.merge(step_cfg, meta)
    overrides_list = list(meta.overrides)
    hydra_cfg = _compose_with_existing_hydra(overrides_list)

    OmegaConf.set_struct(hydra_cfg, False)
    step_cfg = OmegaConf.merge(hydra_cfg, meta)
    step_cfg.parent_id = full_cfg.attempt_id
    OmegaConf.set_struct(step_cfg, True)
    OmegaConf.resolve(step_cfg)

    return step_cfg


class StepConfigManager:
    """Receives a fully composed DictConfig for a single step, creates
    the necessary directory structure, and writes a resolved copy of
    the configuration next to the step files.
    """

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "StepConfigManager":
        return cls(cfg)

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        print_config_tree(cfg)
        self._resolved = OmegaConf.to_container(cfg, resolve=True)

        self.base_dir: Path
        self.exp_dir: Path
        self.step_dir: Path
        self.log_dir: Path
        self.ckpt_dir: Path
        self.artifact_dir: Path
        self.config_path: Path

        self._prepare_paths()
        self._dump_config()

    @property
    def paths(self) -> DictConfig:
        return self._cfg.paths

    def _prepare_paths(self) -> None:
        """Creates the folder structure for one step and a global log view.

        Real folders
        ------------
        <experiment_dir>/<step_name>/logs
        <experiment_dir>/logs/

        Symlink
        -------
        <experiment_dir>/logs/<step_name> -> <experiment_dir>/<step_name>/logs
        """
        p = self.paths

        self.base_dir = Path(p.base_dir).expanduser()
        self.exp_dir = Path(p.experiment_dir).expanduser()

        step_name = self._cfg.step_name
        self.step_dir = self.exp_dir / step_name

        self.log_dir = self.step_dir / "logs"
        self.ckpt_dir = self.step_dir / "checkpoints"
        self.artifact_dir = self.step_dir / "artifacts"

        self.global_log_dir = self.exp_dir / "logs"

        for d in [
            self.base_dir,
            self.exp_dir,
            self.step_dir,
            self.log_dir,
            self.ckpt_dir,
            self.artifact_dir,
            self.global_log_dir,
        ]:
            d.mkdir(mode=0o777, parents=True, exist_ok=True)

        unified_link = self.global_log_dir / step_name
        try:
            if unified_link.exists() or unified_link.is_symlink():
                unified_link.unlink()
            unified_link.symlink_to(self.log_dir, target_is_directory=True)
        except OSError as e:
            log.warning(f"Could not create global log symlink {unified_link} -> {self.log_dir}: {e}")

        self._cfg.paths.experiment_dir = str(self.step_dir)
        self._cfg.paths.base_artifacts_dir = str(self.artifact_dir)
        self._cfg.paths.base_checkpoint_dir = str(self.ckpt_dir)
        self._cfg.paths.log_dir = str(self.log_dir)

        self.config_path = self.step_dir / f"{self._cfg.job_name}_{step_name}_resolved.yaml"

    def _dump_config(self) -> None:
        OmegaConf.save(self._cfg, self.config_path, resolve=True)
        log.info(f"Successfully dumped to {self.config_path}")
