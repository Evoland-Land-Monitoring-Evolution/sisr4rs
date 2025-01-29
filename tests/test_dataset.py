#!/usr/bin/env python
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
"""
Tests for dataset module
"""
import os
from typing import Any

import pytest
import torch
from sensorsio.sentinel2 import Sentinel2

from torchsisr import dataset
from torchsisr.custom_types import BatchData

if "SEN2VENUS_PATH" in os.environ:
    SEN2VENUS_DATASET_PATH = os.environ["SEN2VENUS_PATH"]
else:
    SEN2VENUS_DATASET_PATH = "UNDEFINED"

if "WORLDSTRATX4_PATH" in os.environ:
    WORLDSTRATX4_DATASET_PATH = os.environ["WORLDSTRATX4_PATH "]
else:
    WORLDSTRATX4_DATASET_PATH = "UNDEFINED"

if "WORLDSTRATX2_PATH" in os.environ:
    WORLDSTRATX2_DATASET_PATH = os.environ["WORLDSTRATX2_PATH "]
else:
    WORLDSTRATX2_DATASET_PATH = "UNDEFINED"


def test_match_bands():
    """
    Test the match_band function
    """
    target_bands = tuple(Sentinel2.GROUP_10M)
    source_bands = (Sentinel2.B2, Sentinel2.B5)

    found, missing = dataset.match_bands(source_bands, target_bands)

    assert found == (0,)
    assert missing == (1, 2, 3)

    source_bands = (Sentinel2.Band("B2"), Sentinel2.Band("B5"))

    found, missing = dataset.match_bands(source_bands, target_bands)

    assert found == (0,)
    assert missing == (1, 2, 3)


@pytest.fixture(
    name="single_dataset_factory",
    params=[
        dataset.Sen2VnsSingleSiteDatasetConfig(True, False, False),
        dataset.Sen2VnsSingleSiteDatasetConfig(True, True, False),
        dataset.Sen2VnsSingleSiteDatasetConfig(True, True, True),
        dataset.Sen2VnsSingleSiteDatasetConfig(False, True, False),
        dataset.Sen2VnsSingleSiteDatasetConfig(False, True, True),
    ],
)
def fixture_single_dataset_factory(request) -> dataset.Sen2VnsSingleSiteDataset:
    """
    Factory for single dataset configuration
    """
    config: dataset.Sen2VnsSingleSiteDatasetConfig = request.param
    site_path = os.path.join(SEN2VENUS_DATASET_PATH, "ARM")

    return dataset.Sen2VnsSingleSiteDataset(site_path, config)


@pytest.mark.requires_dataset
def test_single_site_dataset_shape(single_dataset_factory: Any) -> None:
    """
    Test the single site dataset
    """
    ds = single_dataset_factory

    assert len(ds) == 15859  # Amount of ARM patches

    # Get first patch
    sample: BatchData = ds[0]

    # Check shapes
    if ds.config.load_10m_data:
        assert sample.network_input.hr_tensor.shape[0] == 4
        assert sample.network_input.hr_tensor.shape[1] == 128
        assert sample.network_input.hr_tensor.shape[2] == 128
        assert sample.target.shape[1] == 256
        assert sample.target.shape[2] == 256
        assert sample.network_input.hr_bands == (
            Sentinel2.B2,
            Sentinel2.B3,
            Sentinel2.B4,
            Sentinel2.B8,
        )
        if ds.config.load_20m_data:
            assert sample.target.shape[0] == 8
            assert sample.network_input.lr_tensor is not None
            assert sample.network_input.lr_tensor.shape[1] == 64
            assert sample.network_input.lr_tensor.shape[2] == 64
            assert sample.target_bands == (
                Sentinel2.B2,
                Sentinel2.B3,
                Sentinel2.B4,
                Sentinel2.B8,
                Sentinel2.B5,
                Sentinel2.B6,
                Sentinel2.B7,
                Sentinel2.B8A,
            )

            if ds.config.load_b11b12:
                assert sample.network_input.lr_tensor is not None
                assert sample.network_input.lr_tensor.shape[0] == 6
                assert sample.network_input.lr_bands == (
                    Sentinel2.B5,
                    Sentinel2.B6,
                    Sentinel2.B7,
                    Sentinel2.B8A,
                    Sentinel2.B11,
                    Sentinel2.B12,
                )
            else:
                assert sample.network_input.lr_tensor is not None
                assert sample.network_input.lr_tensor.shape[0] == 4
                assert sample.network_input.lr_bands == (
                    Sentinel2.B5,
                    Sentinel2.B6,
                    Sentinel2.B7,
                    Sentinel2.B8A,
                )
        else:
            assert sample.target.shape[0] == 4
            assert sample.target_bands == (
                Sentinel2.B2,
                Sentinel2.B3,
                Sentinel2.B4,
                Sentinel2.B8,
            )
            assert sample.network_input.lr_tensor is None
    else:
        assert sample.target.shape[0] == 4
        assert sample.target.shape[1] == 256
        assert sample.target.shape[2] == 256
        assert sample.target_bands == (
            Sentinel2.B5,
            Sentinel2.B6,
            Sentinel2.B7,
            Sentinel2.B8A,
        )
        assert sample.network_input.hr_tensor.shape[1] == 64
        assert sample.network_input.hr_tensor.shape[2] == 64

        if ds.config.load_b11b12:
            assert sample.network_input.hr_tensor.shape[0] == 6
            assert sample.network_input.hr_bands == (
                Sentinel2.B5,
                Sentinel2.B6,
                Sentinel2.B7,
                Sentinel2.B8A,
                Sentinel2.B11,
                Sentinel2.B12,
            )

        else:
            assert sample.network_input.hr_tensor.shape[0] == 4
            assert sample.network_input.hr_bands == (
                Sentinel2.B5,
                Sentinel2.B6,
                Sentinel2.B7,
                Sentinel2.B8A,
            )


@pytest.mark.requires_dataset
def test_single_site_dataset_in_dataloader(single_dataset_factory: Any) -> None:
    """
    Test that the single dataset can be put into a dataloader
    """
    ds = single_dataset_factory

    dl = torch.utils.data.DataLoader(
        ds, batch_size=4, num_workers=1, collate_fn=dataset.batch_data_collate_fn
    )

    batch = next(iter(dl))

    assert batch.network_input.hr_tensor.shape[0] == 4
    assert batch.target.shape[0] == 4


@pytest.mark.requires_dataset
def test_wald_batch() -> None:
    """
    Test wald function
    """
    config = dataset.Sen2VnsSingleSiteDatasetConfig(True, True, True)
    site_path = os.path.join(SEN2VENUS_DATASET_PATH, "ARM")
    ds = dataset.Sen2VnsSingleSiteDataset(site_path, config)

    dl = torch.utils.data.DataLoader(
        ds, batch_size=2, num_workers=1, collate_fn=dataset.batch_data_collate_fn
    )
    batch = next(iter(dl))

    print(batch.network_input.hr_tensor.shape)
    sim_wald_batch = dataset.wald_batch(dataset.batch_to_millirefl(batch))
    assert len(sim_wald_batch.target_bands) == 10
    assert sim_wald_batch.target.shape == torch.Size((2, 10, 64, 64))
    assert len(sim_wald_batch.network_input.hr_bands) == 4
    assert sim_wald_batch.network_input.hr_tensor.shape == torch.Size((2, 4, 32, 32))
    assert sim_wald_batch.network_input.lr_tensor is not None
    assert sim_wald_batch.network_input.lr_bands is not None
    assert len(sim_wald_batch.network_input.lr_bands) == 6
    assert sim_wald_batch.network_input.lr_tensor.shape == torch.Size((2, 6, 16, 16))


@pytest.mark.requires_dataset
def test_simulate_batch(single_dataset_factory: Any) -> None:
    """
    Test the batch_simulation function
    """
    ds = single_dataset_factory

    dl = torch.utils.data.DataLoader(
        ds, batch_size=2, num_workers=1, collate_fn=dataset.batch_data_collate_fn
    )

    batch = next(iter(dl))
    batch = dataset.batch_to_millirefl(batch)
    noise_std = torch.full((8,), 0.001)
    sim_batch = dataset.simulate_batch(batch)
    sim_batch = dataset.simulate_batch(batch, noise_std=noise_std)

    assert (
        sim_batch.network_input.hr_tensor.shape == batch.network_input.hr_tensor.shape
    )

    if batch.network_input.lr_tensor is not None:
        assert (
            sim_batch.network_input.lr_tensor is not None
            and sim_batch.network_input.lr_tensor.shape
            == batch.network_input.lr_tensor.shape
        )


@pytest.mark.requires_dataset
def test_single_site_invalid_config() -> None:
    """
    Test that we can not build invalid configurations
    """

    site_path = os.path.join(SEN2VENUS_DATASET_PATH, "ARM")
    with pytest.raises(NotImplementedError):
        dataset.Sen2VnsSingleSiteDataset(
            site_path, dataset.Sen2VnsSingleSiteDatasetConfig(False, False, False)
        )
        dataset.Sen2VnsSingleSiteDataset(
            site_path, dataset.Sen2VnsSingleSiteDatasetConfig(False, False, True)
        )
        dataset.Sen2VnsSingleSiteDataset(
            site_path, dataset.Sen2VnsSingleSiteDatasetConfig(True, False, True)
        )


@pytest.fixture(
    name="multi_dataset_factory",
    params=[
        dataset.Sen2VnsSingleSiteDatasetConfig(True, False, False),
        dataset.Sen2VnsSingleSiteDatasetConfig(True, True, False),
        dataset.Sen2VnsSingleSiteDatasetConfig(True, True, True),
        dataset.Sen2VnsSingleSiteDatasetConfig(False, True, False),
        dataset.Sen2VnsSingleSiteDatasetConfig(False, True, True),
    ],
)
def fixture_multi_dataset_factory(request: Any) -> dataset.Sen2VnsMultiSiteDataset:
    """
    Factory for single dataset configuration
    """
    config: dataset.Sen2VnsSingleSiteDatasetConfig = request.param

    return dataset.Sen2VnsMultiSiteDataset(
        SEN2VENUS_DATASET_PATH, ["ARM", "FR-LQ1"], config
    )


@pytest.mark.requires_dataset
def test_multi_site_dataset(multi_dataset_factory: Any) -> None:
    """
    Test multi-site dataset
    """
    ds = multi_dataset_factory
    assert ds[0]


@pytest.fixture(
    name="datamodule_factory",
    params=[
        dataset.Sen2VnsDataModuleConfig(
            dataset_folder=SEN2VENUS_DATASET_PATH,
            testing_sites=["ARM"],
            num_workers=1,
            single_site_config=dataset.Sen2VnsSingleSiteDatasetConfig(
                True, False, False
            ),
        )
    ],
)
def fixture_datamodule_factory(request: Any) -> dataset.Sen2VnsDataModule:
    """
    Generate datamodule from config
    """
    config = request.param

    dm = dataset.Sen2VnsDataModule(config)

    return dm


@pytest.mark.requires_dataset
def test_datamodule(datamodule_factory: dataset.Sen2VnsDataModule) -> None:
    """
    Test datamodule
    """
    train_dataloader = datamodule_factory.train_dataloader()
    val_dataloader = datamodule_factory.val_dataloader()
    test_dataloader = datamodule_factory.test_dataloader()

    for dl in (train_dataloader, val_dataloader, test_dataloader):
        next(iter(dl))


def test_worldstrat_dataset():
    """
    Test the worldstrat dataset
    """
    ds = dataset.WorldStratDataset(os.path.join(WORLDSTRATX4_DATASET_PATH, "index.csv"))

    batch = ds[0]

    assert batch.network_input.hr_tensor.shape == torch.Size([4, 128, 128])
    assert batch.target.shape == torch.Size([4, 512, 512])
    assert batch.network_input.lr_tensor is None


def test_worldstrat_datamodule():
    """
    Test the worldstrat dataset
    """
    dm = dataset.WorldStratDataModule(
        os.path.join(WORLDSTRATX2_DATASET_PATH, "index.csv"),
        num_workers=1,
        batch_size=2,
        testing_validation_batch_size=2,
    )

    dlt = dm.train_dataloader()
    dlv = dm.val_dataloader()
    dltest = dm.test_dataloader()

    next(iter(dlt))
    next(iter(dlv))
    next(iter(dltest))
