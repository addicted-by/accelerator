import warnings
from collections.abc import Sequence
from pathlib import Path

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from accelerator.utilities.distributed_state.state import distributed_state


@distributed_state.on_main_process
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "pipeline",
        "model",
        "training_components",
        "callbacks",
        "datamodule",
        "acceleration",
        "paths",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
    ignore: Sequence[str] = ["hydra"],
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config
            components are printed.
        resolve (bool, optional): Whether to resolve reference fields of
            DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra
            output folder.
        ignore (Sequence[str]): Which nodes and the corresponding sub-graphs to ignore.

    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else print(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue and field not in ignore:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)
    # print("YEBAT")
    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.experiment_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@distributed_state.on_main_process
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in
    config.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        print("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        print(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.experiment_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing

    Args:
        cfg (DictConfig): Main config.

    """
    # return if no `extras` config
    if not cfg.get("extras"):
        print("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        print("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        print("Enforcing tags! <cfg.extras.enforce_tags=True>")
        enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        print("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)
