from pathlib import Path
from typing import Optional

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf


def _compose_with_existing_hydra(
    overrides: list[str],
    *,
    conf_root: Optional[Path] = None,
    base_cfg: str = "main",
    version_base: Optional[str] = None,
    return_hydra_config: bool = False,
) -> DictConfig:
    """
    Compose a DictConfig with `overrides` whether or not Hydra is already
    initialised.

    Parameters
    ----------
    overrides : list[str]
        Dot-list overrides to apply, e.g. ["model=resnet50", "lr=0.01"].
    conf_root : Path | None
        Path to your conf/ directory **if Hydra is not already running**.
        Ignored when running inside @hydra.main.
    base_cfg  : str
        Name of the root yaml to use when Hydra is *not* initialised AND the
        original config name cannot be recovered.

    Returns
    -------
    DictConfig
    """
    if hydra.core.hydra_config.HydraConfig.initialized():
        cli_overrides = list(hydra.core.hydra_config.HydraConfig.get().overrides.task)
        job_cfg_name = HydraConfig.get().job.config_name or base_cfg
        return hydra.compose(
            config_name=job_cfg_name,
            overrides=cli_overrides + overrides,
            return_hydra_config=return_hydra_config,
        )

    if conf_root is None:
        raise ValueError("conf_root must be given when Hydra is not yet initialised.")
    with initialize_config_dir(
        config_dir=str(conf_root), job_name="_compose_with_existing_hydra", version_base=version_base
    ):
        return compose(
            config_name=base_cfg,
            overrides=overrides,
            return_hydra_config=return_hydra_config,
        )


def find_get_objects(cfg):
    """
    Recursively processes a configuration to replace dictionaries containing '_object_' with their corresponding objects.

    Args:
        cfg: The configuration object (dict, list, or other type).

    Returns:
        The processed configuration, with '_object_' dictionaries replaced by objects.
    """
    if isinstance(cfg, (dict, DictConfig)):
        if "_object_" in cfg:
            return hydra.utils.get_object(cfg["_object_"])
        elif "_method_" in cfg:
            return hydra.utils.get_method(cfg["_method_"])
        return {key: find_get_objects(value) for key, value in cfg.items()}
    elif isinstance(cfg, list):
        return [find_get_objects(item) for item in cfg]
    else:
        return cfg


def instantiate_wrapper(cfg, *args, **kwargs):
    """
    Processes the configuration and instantiates it using Hydra's instantiate utility.

    Args:
        cfg: The configuration object to process and instantiate.
        *args: Positional arguments passed to hydra.utils.instantiate.
        **kwargs: Keyword arguments passed to hydra.utils.instantiate.

    Returns:
        The instantiated object or processed configuration, or None if cfg is None.
    """
    if cfg is None:
        return None

    processed_cfg = find_get_objects(cfg)
    if isinstance(processed_cfg, (dict, list, DictConfig, ListConfig)):
        return hydra.utils.instantiate(processed_cfg, *args, **kwargs)
    else:
        return processed_cfg


def instantiate(cfg, *args, **kwargs):
    """
    A convenience wrapper for instantiating a configuration.

    Args:
        cfg: The configuration object to instantiate.
        *args: Positional arguments passed to instantiate_wrapper.
        **kwargs: Keyword arguments passed to instantiate_wrapper.

    Returns:
        The instantiated object or processed configuration.
    """
    return instantiate_wrapper(cfg, *args, **kwargs)


def load_sub_configs(cfg, *args, **kwargs):
    """
    Recursively processes a configuration object, loading sub-configurations specified by '_load_' keys
    and applying overrides specified by '_override_' keys.

    Args:
        cfg: The configuration object (dict or list, typically an OmegaConf DictConfig or ListConfig).
        *args: Additional positional arguments (unused, kept for compatibility).
        **kwargs: Additional keyword arguments (unused, kept for compatibility).

    Returns:
        The processed configuration object.
    """
    original_struct = OmegaConf.is_struct(cfg)

    if original_struct:
        OmegaConf.set_struct(cfg, False)

    if isinstance(cfg, (dict, DictConfig)):
        overrides = cfg.get("_override_", {}).copy()

        if "_load_" in cfg:
            loaded_cfg = OmegaConf.load(cfg["_load_"])
            cfg.clear()  # Clear current dict, including '_load_'
            for key, value in loaded_cfg.items():
                OmegaConf.update(cfg, key, value, force_add=True)
            # cfg.update(loaded_cfg)  # Update with loaded config

        for key, value in overrides.items():
            OmegaConf.update(cfg, key, value)

        for key, value in cfg.items():
            if isinstance(value, (dict, list, DictConfig)):
                cfg[key] = load_sub_configs(value, *args, **kwargs)

    elif isinstance(cfg, list):
        for i, item in enumerate(cfg):
            if isinstance(item, (dict, list)):
                cfg[i] = load_sub_configs(item, *args, **kwargs)

    if original_struct:
        OmegaConf.set_struct(cfg, True)

    return cfg


def resolve(cfg):
    if cfg is None:
        return None

    cfg = load_sub_configs(cfg)
    return OmegaConf.resolve(cfg)


def jupyter_overrides(job_name: str = "notebook_playground"):
    overrides = [
        f"{job_name=}",
        f"hydra.job.name={job_name}",
        "hydra.runtime.cwd=${oc.env:PWD}",
        "hydra.runtime.output_dir=./notebook_runs/${now:%Y-%m-%d_%H-%M-%S}",
    ]

    return overrides
