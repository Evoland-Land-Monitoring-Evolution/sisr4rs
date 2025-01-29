#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from torchsisr import dataset, hydra_utils


def compute_gradient_magnitude(data: torch.Tensor) -> torch.Tensor:
    """ """
    # Compute gradient magnitude
    grad_x, grad_y = torch.gradient(data, dim=(-1, -2))

    # Accumulate accross all bands
    grad_x = grad_x.mean(dim=1, keepdim=True)
    grad_y = grad_y.mean(dim=1, keepdim=True)

    return torch.sqrt(grad_x**2 + grad_y**2).flatten()


@hydra.main(version_base=None, config_path="../hydra/", config_name="main.yaml")
def main(config: DictConfig):
    # Call extras
    hydra_utils.extras(config)

    # load samples
    single_site_config = hydra.utils.instantiate(
        config.datamodule.single_site_config, load_b11b12=True
    )
    datamodule_config = hydra.utils.instantiate(
        config.datamodule.config, single_site_config=single_site_config
    )

    # First, run on training set
    training_dataloader = dataset.Sen2VnsDataModule(
        datamodule_config
    ).train_dataloader()

    # Estimate on that amount on batch
    nb_batches = 100

    iter_dataloader = iter(training_dataloader)

    grad_mag_samples = []
    source_10_samples = []
    source_20_samples = []
    target_samples = []

    for _ in tqdm(range(nb_batches), total=nb_batches, desc="Collecting samples"):
        current_batch = next(iter_dataloader)

        source_10_samples.append(current_batch.network_input.hr_tensor)
        source_20_samples.append(current_batch.network_input.lr_tensor)
        target_samples.append(current_batch.target)
        grad_mag_samples.append(
            compute_gradient_magnitude(current_batch.target[:, :8, ...] / 10000.0)
        )

    source_10_samples = torch.cat(source_10_samples)
    source_20_samples = torch.cat(source_20_samples)
    target_samples = torch.cat(target_samples)

    print(f"10m bands mean: {(source_10_samples/10000.).mean(dim=(0,2,3))}")
    print(f"10m bands std: {(source_10_samples/10000.).std(dim=(0,2,3))}")
    print(f"20m bands mean: {(source_20_samples/10000.).mean(dim=(0,2,3))}")
    print(f"20m bands std: {(source_20_samples/10000.).std(dim=(0,2,3))}")
    print(f"Target bands mean: {(target_samples/10000.).mean(dim=(0,2,3))}")
    print(f"Target bands std: {(target_samples/10000.).std(dim=(0,2,3))}")

    grad_mag_samples = torch.cat(grad_mag_samples)

    low_strata_min_percent = 0
    low_strata_max_percent = 25

    high_strata_min_percent = 75
    high_strata_max_percent = 100

    low_strata_min = np.percentile(
        grad_mag_samples.cpu().detach().numpy(), low_strata_min_percent
    )
    low_strata_max = np.percentile(
        grad_mag_samples.cpu().detach().numpy(), low_strata_max_percent
    )
    high_strata_min = np.percentile(
        grad_mag_samples.cpu().detach().numpy(), high_strata_min_percent
    )
    high_strata_max = np.percentile(
        grad_mag_samples.cpu().detach().numpy(), high_strata_max_percent
    )

    print(f"{low_strata_min=}, {low_strata_max=}")
    print(f"{high_strata_min=}, {high_strata_max=}")


if __name__ == "__main__":
    main()
