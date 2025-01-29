#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
A tool to generate samples for training of super-resolution algorithms
"""


import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from torchsisr import dataset, hydra_utils


@hydra.main(version_base=None, config_path="../hydra/", config_name="main.yaml")
def main(config: DictConfig):
    """
    Hydra main function
    """
    # Call extras
    hydra_utils.extras(config)

    # load samples
    datamodule_config = hydra.utils.instantiate(
        config.datamodule.config, num_workers=0, prefetch_factor=None
    )

    val_dl = dataset.Sen2VnsDataModule(datamodule_config).val_dataloader()

    for _ in tqdm(iter(val_dl), total=len(val_dl), desc="first pass"):
        pass

    for _ in tqdm(iter(val_dl), total=len(val_dl), desc="second pass"):
        pass


if __name__ == "__main__":
    main()
