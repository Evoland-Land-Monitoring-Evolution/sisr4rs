# From https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/__init__.py
"""
A few hydra related utilities
"""

import logging
import warnings
from collections.abc import Sequence

import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.
    Utilities:
     - Ignoring python warnings
     - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        logging.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

        # pretty print config tree using Rich library if <config.print_config=True>
        if config.get("print_config"):
            logging.info("Printing config tree with Rich! <config.print_config=True>")
            print_config(config, resolve=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "trainer",
        "callbacks",
        "loggers",
        "log_dir",
    ),
    resolve: bool = True,
) -> None:
    """

    Prints content of DictConfig using Rich library and its tree structure.
       Args:
            config (DictConfig): Configuration composed by Hydra.

        print_order (Sequence[str], optional): Determines in what
        order config components are printed.

    resolve (bool, optional): Whether to resolve reference fields of
    DictConfig.

    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee: list[str] = []

    for field in print_order:
        if field in config:
            quee.append(field)
        else:
            logging.info("Field '%s' not found in config", field)

    for config_field in config.keys():
        if config_field not in quee:
            quee.append(str(config_field))

    for config_field in quee:
        branch = tree.add(config_field, style=style, guide_style=style)

        config_group = config[config_field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w", encoding="utf-8") as file:
        rich.print(tree, file=file)
