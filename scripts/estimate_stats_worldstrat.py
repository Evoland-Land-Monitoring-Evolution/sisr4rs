#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from torchsisr import hydra_utils


@hydra.main(version_base=None, config_path="../hydra/", config_name="main.yaml")
def main(config: DictConfig):
    # Call extras
    hydra_utils.extras(config)

    # First, run on training set
    training_dataloader = hydra.utils.instantiate(
        config.datamodule.data_module
    ).train_dataloader()

    # Estimate on that amount on batch
    nb_batches = 100

    iter_dataloader = iter(training_dataloader)
    source_10_samples = []
    target_samples = []

    for _ in tqdm(range(nb_batches), total=nb_batches, desc="Collecting samples"):
        current_batch = next(iter_dataloader)

        source_10_samples.append(current_batch.network_input.hr_tensor)
        target_samples.append(current_batch.target)

    source_10_samples = torch.cat(source_10_samples)

    target_samples = torch.cat(target_samples)

    print(f"10m bands mean: {(source_10_samples/10000.).mean(dim=(0,2,3))}")
    print(f"10m bands std: {(source_10_samples/10000.).std(dim=(0,2,3))}")
    print(f"Target bands mean: {(target_samples/10000.).mean(dim=(0,2,3))}")
    print(f"Target bands std: {(target_samples/10000.).std(dim=(0,2,3))}")


if __name__ == "__main__":
    main()
