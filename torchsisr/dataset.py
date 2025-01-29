# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains helper classes related to the sen2venµs dataset
"""

import glob
import logging
import math
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio as rio
import torch
from sensorsio.sentinel2 import Sentinel2

from torchsisr.custom_types import BatchData, NetworkInput


def generate_psf_kernel(
    res: float, mtf_res: float, mtf_fc: float, half_kernel_width: int | None = None
) -> np.ndarray:
    """
    Generate a psf convolution kernel
    """
    fc = 0.5 / mtf_res
    sigma = math.sqrt(-math.log(mtf_fc) / 2) / (math.pi * fc)
    if half_kernel_width is None:
        half_kernel_width = int(math.ceil(mtf_res / (res)))
    kernel = np.zeros((2 * half_kernel_width + 1, 2 * half_kernel_width + 1))
    for i in range(0, half_kernel_width + 1):
        for j in range(0, half_kernel_width + 1):
            dist = res * math.sqrt(i**2 + j**2)
            psf = np.exp(-(dist * dist) / (2 * sigma * sigma)) / (
                sigma * math.sqrt(2 * math.pi)
            )
            kernel[half_kernel_width - i, half_kernel_width - j] = psf
            kernel[half_kernel_width - i, half_kernel_width + j] = psf
            kernel[half_kernel_width + i, half_kernel_width + j] = psf
            kernel[half_kernel_width + i, half_kernel_width - j] = psf

    return (kernel / np.sum(kernel)).astype(np.float32)


def generic_downscale(
    data: torch.Tensor,
    factor: float = 2.0,
    mtf: float = 0.1,
    padding="same",
    mode: str = "bicubic",
    hkw: int = 3,
):
    """
    Downsample patches with proper aliasing filtering
    """
    # Generate psf kernel for target MTF
    psf_kernel = torch.tensor(
        generate_psf_kernel(1.0, factor, mtf, hkw), device=data.device, dtype=data.dtype
    )
    # Convolve data with psf kernel
    data = torch.nn.functional.pad(data, (hkw, hkw, hkw, hkw), mode="reflect")
    # pylint: disable=not-callable
    data = torch.nn.functional.conv2d(
        data,
        psf_kernel[None, None, :, :].expand(data.shape[1], -1, -1, -1),
        groups=data.shape[1],
        padding=padding,
    )
    # Downsample with nearest neighbors
    data = torch.nn.functional.interpolate(data, scale_factor=1 / factor, mode=mode)
    return data


def downscale_venus(
    patches: torch.Tensor,
    bands: tuple[Sentinel2.Band, ...],
    resolution: float = 10.0,
    half_kernel_width=5,
    noise_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Downscale Venµs patch at Sentinel2 resolution, in 3 steps:
    1. Spatial convolution by a gaussian kernel tuned to Sentinel2 bands MTF value
    2. Decimation
    3. Add optional gaussian noise

    :param t: The tensor witht eh band to downscale (shape [n,c,w,h])
    :param bands: The corresponding Sentinel2 bands
    :param resolution: The target resolution
    :param half_kernel_width: half width of the convolution kernel
    :param noise_std standard deviation of the noise to add (or None to disable)
    :return: Tensor of downscaled patches
    """
    pad = torch.nn.ReflectionPad2d(half_kernel_width)
    s2_psf = torch.tensor(
        Sentinel2.generate_psf_kernel(
            list(bands), resolution=5.0, half_kernel_width=half_kernel_width
        ),
        device=patches.device,
        dtype=patches.dtype,
    )
    # pylint: disable=not-callable
    filtered_tensor = torch.nn.functional.conv2d(
        pad(patches), s2_psf[:, None, :, :], groups=patches.shape[1], padding="valid"
    )
    res = torch.nn.functional.interpolate(
        filtered_tensor, scale_factor=5.0 / resolution, mode="bicubic"
    )

    if noise_std is not None:
        res += torch.normal(
            0.0,
            noise_std[None, :, None, None].expand(
                (res.shape[0], -1, res.shape[2], res.shape[3])
            ),
        )

    return res


# Cache the results of this function since it will be called many time with same args
@lru_cache
def match_bands(
    src_bands: tuple[Sentinel2.Band], target_bands: tuple[Sentinel2.Band]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Find matching between list of bands
    """
    out = tuple(target_bands.index(b) for b in src_bands if b in target_bands)
    out_missing = tuple(
        target_bands.index(b) for b in target_bands if b not in src_bands
    )
    return out, out_missing


def batch_to_millirefl(
    batch: BatchData, dtype=torch.float32, scale: float = 10000.0
) -> BatchData:
    """
    Scale batch to millirefl
    """
    hr_tensor = (batch.network_input.hr_tensor / scale).to(dtype=dtype)
    target = (batch.target / scale).to(dtype=dtype)
    lr_tensor: torch.Tensor | None = None
    if batch.network_input.lr_tensor is not None:
        lr_tensor = (batch.network_input.lr_tensor / scale).to(dtype=dtype)

    return BatchData(
        NetworkInput(
            hr_tensor,
            batch.network_input.hr_bands,
            lr_tensor,
            batch.network_input.lr_bands,
        ),
        target,
        batch.target_bands,
    )


def flip_concat(data: torch.Tensor) -> torch.Tensor:
    """
    Make a 4x4 mosaic with flipped axes
    """
    data = torch.cat((data, data.flip(-2)), dim=-2)
    data = torch.cat((data, data.flip(-1)), dim=-1)
    return data


def align_min_max(data: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Linear scaling of data so that it aligns on ref min / max per band on each patch
    """
    assert data.shape[0] == ref.shape[0]
    assert data.shape[1] == ref.shape[1]

    min_ref_v = ref.amin(dim=(2, 3))
    max_ref_v = ref.amax(dim=(2, 3))
    min_data_v = data.amin(dim=(2, 3))
    max_data_v = data.amax(dim=(2, 3))

    slope = (max_ref_v - min_ref_v) / (max_data_v - min_data_v)

    output = (
        min_ref_v[:, :, None, None]
        + (data - min_data_v[:, :, None, None]) * slope[:, :, None, None]
    )
    return output


def align_min_max_batch(batch: BatchData) -> BatchData:
    """
    Align target min max with input data stats
    """
    required_bands, _ = match_bands(batch.network_input.hr_bands, batch.target_bands)

    target_hr_aligned = align_min_max(
        batch.target[:, list(required_bands), ...], batch.network_input.hr_tensor
    )

    if batch.network_input.lr_tensor is not None:
        required_bands, _ = match_bands(
            batch.network_input.lr_bands, batch.target_bands
        )
        target_lr_aligned = align_min_max(
            batch.target[:, list(required_bands), ...],
            batch.network_input.lr_tensor[:, :4, ...],
        )
        target = torch.cat((target_hr_aligned, target_lr_aligned), dim=1)
    else:
        target = target_hr_aligned
    return BatchData(batch.network_input, target, batch.target_bands)


def high_pass_filtering(
    data: torch.Tensor, mtf: float, scale_factor: float
) -> torch.Tensor:
    """
    Perform high pass filtering
    """
    additional_crop = int(
        data.shape[-1] - (scale_factor * np.floor(data.shape[-1] / scale_factor))
    )

    if additional_crop > 0:
        data = data[:, :, additional_crop:, additional_crop:]

    data_hf = data - torch.nn.functional.interpolate(
        generic_downscale(data, factor=scale_factor, mtf=mtf, padding="valid"),
        scale_factor=scale_factor,
        align_corners=False,
        mode="bicubic",
    )

    return data_hf


def wald_batch(
    batch: BatchData,
    noise_std: torch.Tensor | None = None,
    pad_to_input_size: bool = False,
    mtf: float = 0.1,
) -> BatchData:
    """
    Apply Wald protocol to simulate a downscaled batch
    """
    # Only support the case where we have all bands
    assert batch.network_input.hr_bands == tuple(Sentinel2.GROUP_10M)
    assert (
        batch.network_input.lr_bands is not None
        and batch.network_input.lr_bands == tuple(Sentinel2.GROUP_20M)
    )
    assert batch.network_input.lr_tensor is not None

    target_tensor = torch.cat(
        (
            generic_downscale(
                batch.network_input.hr_tensor, factor=2.0, padding="valid", mtf=mtf
            ),
            batch.network_input.lr_tensor,
        ),
        dim=1,
    )
    if pad_to_input_size:
        target_tensor = flip_concat(target_tensor)
    hr_tensor = generic_downscale(
        batch.network_input.hr_tensor,
        factor=4.0,
        padding="valid",
        mtf=mtf,
    )
    if pad_to_input_size:
        hr_tensor = flip_concat(hr_tensor)
    lr_tensor = generic_downscale(
        batch.network_input.lr_tensor, factor=4.0, padding="valid", mtf=mtf
    )
    if pad_to_input_size:
        lr_tensor = flip_concat(lr_tensor)

    if noise_std is not None:
        hr_tensor += torch.normal(
            0.0,
            noise_std[None, :4, None, None].expand(
                (hr_tensor.shape[0], -1, hr_tensor.shape[2], hr_tensor.shape[3])
            ),
        )
        lr_tensor += torch.normal(
            0.0,
            noise_std[None, 4:, None, None].expand(
                (lr_tensor.shape[0], -1, lr_tensor.shape[2], lr_tensor.shape[3])
            ),
        )

    return BatchData(
        NetworkInput(
            hr_tensor, tuple(Sentinel2.GROUP_10M), lr_tensor, tuple(Sentinel2.GROUP_20M)
        ),
        target_tensor,
        tuple(Sentinel2.GROUP_10M + Sentinel2.GROUP_20M),
    )


def simulate_batch(
    batch: BatchData, noise_std: torch.Tensor | None = None, mtf: float = 0.1
) -> BatchData:
    """
    Create a simulated batch from a real one by downscaling target
    """
    assert len(batch.network_input.hr_tensor.shape) == 4
    assert noise_std is None or noise_std.shape[0] == batch.target.shape[1]

    # Resolve bands matching
    required_bands, _ = match_bands(batch.network_input.hr_bands, batch.target_bands)
    _, missing_bands = match_bands(batch.target_bands, batch.network_input.hr_bands)

    factor = batch.target.shape[-1] / batch.network_input.hr_tensor.shape[-1]
    # Perform downscaling
    simulated_hr_input = generic_downscale(
        batch.target[:, list(required_bands), ...],
        factor=factor,
        mtf=mtf,
        padding="valid",
    )
    if noise_std is not None:
        noise_std = noise_std[list(required_bands)]
        simulated_hr_input += torch.normal(
            0.0,
            noise_std[None, :, None, None].expand(
                (
                    simulated_hr_input.shape[0],
                    -1,
                    simulated_hr_input.shape[2],
                    simulated_hr_input.shape[3],
                )
            ),
        )

    if missing_bands:
        simulated_hr_input = torch.cat(
            (
                simulated_hr_input,
                batch.network_input.hr_tensor[:, missing_bands, ...],
            ),
            dim=1,
        )

    # Same code for simulating LR input if required
    simulated_lr_input: torch.Tensor | None = None
    if (
        batch.network_input.lr_tensor is not None
        and batch.network_input.lr_bands is not None
    ):
        lr_required_bands, _ = match_bands(
            batch.network_input.lr_bands, batch.target_bands
        )
        _, lr_missing_bands = match_bands(
            batch.target_bands, batch.network_input.lr_bands
        )
        factor = batch.target.shape[-1] / batch.network_input.lr_tensor.shape[-1]
        simulated_lr_input = generic_downscale(
            batch.target[:, list(lr_required_bands), ...],
            factor=factor,
            mtf=mtf,
            padding="valid",
        )
        assert simulated_lr_input is not None

        if noise_std is not None:
            noise_std = noise_std[list(required_bands)]
            simulated_lr_input += torch.normal(
                0.0,
                noise_std[None, :, None, None].expand(
                    (
                        simulated_lr_input.shape[0],
                        -1,
                        simulated_lr_input.shape[2],
                        simulated_lr_input.shape[3],
                    )
                ),
            )

        if lr_missing_bands:
            simulated_lr_input = torch.cat(
                (
                    simulated_lr_input,
                    batch.network_input.lr_tensor[:, lr_missing_bands, ...],
                ),
                dim=1,
            )
    return BatchData(
        NetworkInput(
            simulated_hr_input,
            batch.network_input.hr_bands,
            simulated_lr_input,
            batch.network_input.lr_bands,
        ),
        batch.target,
        batch.target_bands,
    )


@dataclass(frozen=True)
class Sen2VnsSingleSiteDatasetConfig:
    """
    Parameters of dataset
    """

    load_10m_data: bool
    load_20m_data: bool
    load_b11b12: bool


class Sen2VnsSingleSiteDataset(torch.utils.data.Dataset):
    """
    A map-style dataset handling a single sen2venµs site
    """

    def __init__(
        self,
        site_path: str,
        config: Sen2VnsSingleSiteDatasetConfig,
    ):
        """ """
        super().__init__()

        index_csv = os.path.join(site_path, "index.csv")
        self.patches_df = pd.read_csv(index_csv, sep="\t")
        self.site_path = site_path
        self.config = config

        # Prevent the build of invalid configurations
        if not self.config.load_10m_data and not self.config.load_20m_data:
            raise NotImplementedError()

        if not self.config.load_20m_data and self.config.load_b11b12:
            raise NotImplementedError()

        # Static tuples for bands and keys
        self.target_bands: tuple[Sentinel2.Band, ...]
        self.source1_bands: tuple[Sentinel2.Band, ...] | None = None
        self.source2_bands: tuple[Sentinel2.Band, ...] | None = None
        self.target_keys: list[str]
        self.source1_keys: list[str]
        self.source2_keys: list[str] | None = None

        if self.config.load_10m_data:
            self.source1_keys = ["b2b3b4b8_10m"]
            self.source1_bands = (
                Sentinel2.B2,
                Sentinel2.B3,
                Sentinel2.B4,
                Sentinel2.B8,
            )
            self.target_bands = (Sentinel2.B2, Sentinel2.B3, Sentinel2.B4, Sentinel2.B8)
            self.target_keys = ["b2b3b4b8_05m"]

            if self.config.load_20m_data:
                self.source2_keys = ["b5b6b7b8a_20m"]
                self.source2_bands = (
                    Sentinel2.B5,
                    Sentinel2.B6,
                    Sentinel2.B7,
                    Sentinel2.B8A,
                )
                self.target_keys.append("b5b6b7b8a_05m")
                self.target_bands = tuple(
                    list(self.target_bands)
                    + [Sentinel2.B5, Sentinel2.B6, Sentinel2.B7, Sentinel2.B8A]
                )

                if self.config.load_b11b12:
                    self.source2_bands = tuple(
                        list(self.source2_bands) + [Sentinel2.B11, Sentinel2.B12]
                    )
                    self.source2_keys += ["b11b12_20m"]
        else:
            if self.config.load_20m_data:
                self.source1_bands = (
                    Sentinel2.B5,
                    Sentinel2.B6,
                    Sentinel2.B7,
                    Sentinel2.B8A,
                )
                self.source1_keys = ["b5b6b7b8a_20m"]
                self.target_bands = (
                    Sentinel2.B5,
                    Sentinel2.B6,
                    Sentinel2.B7,
                    Sentinel2.B8A,
                )
                self.target_keys = ["b5b6b7b8a_05m"]
                if self.config.load_b11b12:
                    self.source1_bands = tuple(
                        list(self.source1_bands) + [Sentinel2.B11, Sentinel2.B12]
                    )
                    self.source1_keys += ["b11b12_20m"]
            else:
                raise NotImplementedError()

    def __read_data(self, filepath: str) -> torch.Tensor:
        """
        Read a single image within zip file and return as Tensor
        """
        with rio.open("/vsizip/" + os.path.join(self.site_path, filepath), "r") as df:
            data = df.read()
            assert data.sum() != 0, filepath
            assert data.min() != -10000, filepath
            return torch.from_numpy(data)

    def __len__(self) -> int:
        """ """
        return len(self.patches_df)

    def __getitem__(self, idx: int) -> BatchData:
        """
        Read a single item
        """

        row = self.patches_df.iloc[idx]

        hr_tensor = torch.cat(
            tuple(self.__read_data(row[k]) for k in self.source1_keys), dim=0
        )

        target_tensor = torch.cat(
            tuple(self.__read_data(row[k]) for k in self.target_keys), dim=0
        )

        lr_tensor: torch.Tensor | None = None

        if self.source2_keys is not None:
            lr_tensor = torch.cat(
                tuple(self.__read_data(row[k]) for k in self.source2_keys), dim=0
            )
        assert self.source1_bands
        return BatchData(
            NetworkInput(hr_tensor, self.source1_bands, lr_tensor, self.source2_bands),
            target_tensor,
            self.target_bands,
        )


class Sen2VnsMultiSiteDataset(torch.utils.data.Dataset):
    """
    A dataset that aggregates all single site datasets
    """

    def __init__(
        self,
        dataset_path: str,
        sites: list[str],
        config: Sen2VnsSingleSiteDatasetConfig,
        max_patches_per_site: int | None = None,
    ):
        """ """
        super().__init__()
        single_site_datasets: list[
            Sen2VnsSingleSiteDataset | torch.utils.data.Subset
        ] = [
            Sen2VnsSingleSiteDataset(os.path.join(dataset_path, site), config)
            for site in sites
        ]
        if max_patches_per_site is not None:
            single_site_datasets = [
                torch.utils.data.Subset(
                    d,
                    random.sample(
                        list(range(len(d))), k=min(len(d), max_patches_per_site)
                    ),
                )
                for d in single_site_datasets
            ]
        self.dataset: torch.utils.data.ConcatDataset = torch.utils.data.ConcatDataset(
            single_site_datasets
        )

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int):
        return self.dataset[idx]


class CacheDataset(torch.utils.data.Dataset):
    """
    A dataset that caches every retrieved tensor for later use
    """

    def __init__(self, dataset: torch.utils.data.Dataset):
        """ """
        super().__init__()
        self.dataset = dataset
        self.cache: dict[int, BatchData] = {}

    def __len__(self):
        """ """
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int):
        """ """
        if idx in self.cache:
            return self.cache[idx]

        ret = self.dataset[idx]
        self.cache[idx] = ret
        return ret


def batch_data_collate_fn(data: list[BatchData]):
    """
    Tells DataLoaer how to collate BatchData

    Assuming all samples in data are of same kind
    """
    hr_source = torch.utils.data.default_collate(
        [d.network_input.hr_tensor for d in data]
    )
    lr_source: torch.Tensor | None = None
    if data[0].network_input.lr_tensor is not None:
        lr_source = torch.utils.data.default_collate(
            [d.network_input.lr_tensor for d in data]
        )
    target = torch.utils.data.default_collate([d.target for d in data])

    return BatchData(
        NetworkInput(
            hr_source,
            data[0].network_input.hr_bands,
            lr_source,
            data[0].network_input.lr_bands,
        ),
        target,
        data[0].target_bands,
    )


@dataclass(frozen=True)
class Sen2VnsDataModuleConfig:
    """
    Datamodule config

    :param dataset_folder: Path to the dataset folder
    :param testing_sites: List of site that are kept for testing
    :param batch_size: Size of the baches
    :param sites: Restrict to a list of sites
    :param validation_ratio: Proportion of pairs kept for validation
    """

    dataset_folder: str
    testing_sites: list[str]
    single_site_config: Sen2VnsSingleSiteDatasetConfig
    sites: list[str] | None = None
    max_patches_per_site: int | None = None
    batch_size: int = 32
    testing_validation_batch_size: int = 128
    validation_ratio: float = 0.1
    train_ratio: float = 0.9
    cache_validation_dataset: bool = False
    cache_testing_dataset: bool = False
    num_workers: int = 0
    prefetch_factor: int | None = 4


class WorldStratDataset(torch.utils.data.Dataset):
    """
    A map-style dataset handling a single sen2venµs site
    """

    def __init__(
        self, index_path: str, split: str = "train", min_correlation: float = 0.1
    ):
        """ """
        super().__init__()

        self.patches_df = pd.read_csv(index_path, sep="\t")
        self.patches_df = self.patches_df[self.patches_df.split == split]
        before_filtering = len(self.patches_df)
        self.patches_df = self.patches_df[self.patches_df.correlation > min_correlation]
        after_filtering = len(self.patches_df)
        print(
            f"Filtering removed {before_filtering-after_filtering} \
            patches for split {split} (remaining: {after_filtering})"
        )
        self.dataset_directory = os.path.dirname(index_path)
        self.bands = (
            Sentinel2.B2,
            Sentinel2.B3,
            Sentinel2.B4,
            Sentinel2.B8,
        )

    def __len__(self) -> int:
        """ """
        return len(self.patches_df)

    def __getitem__(self, idx: int) -> BatchData:
        """
        Read a single item
        """
        row = self.patches_df.iloc[idx]

        hr_file = os.path.join(self.dataset_directory, row.hr_img)
        lr_file = os.path.join(self.dataset_directory, row.lr_img)

        with rio.open(hr_file, "r") as ds:
            hr_tensor = torch.tensor(ds.read())

        with rio.open(lr_file, "r") as ds:
            lr_tensor = torch.tensor(ds.read())

        return BatchData(
            NetworkInput(
                hr_tensor=lr_tensor, hr_bands=self.bands, lr_tensor=None, lr_bands=None
            ),
            hr_tensor,
            self.bands,
        )


class Sen2VnsDataModule(pl.LightningDataModule):
    """
    The datamodule based on simplified dataset
    """

    def __init__(self, config: Sen2VnsDataModuleConfig):
        """
        Constructor
        """
        super().__init__()
        # self.save_hyperparameters()

        self.config = config

        # Type annotations
        self.training_dataset: torch.utils.data.Dataset
        self.validation_dataset: torch.utils.data.Dataset
        self.testing_dataset: torch.utils.data.Dataset

        # Auto-detect sites
        self.sites = config.sites
        if self.sites is None:
            self.sites = [
                s.split()[-1]
                for s in glob.glob(os.path.join(config.dataset_folder, "*"))
                if Path(os.path.join(s, "index.csv")).exists()
                and s.split()[-1] not in config.testing_sites
            ]

        # Build datasets
        self.testing_dataset = Sen2VnsMultiSiteDataset(
            config.dataset_folder,
            config.testing_sites,
            config.single_site_config,
            max_patches_per_site=config.max_patches_per_site,
        )

        remaining_dataset = Sen2VnsMultiSiteDataset(
            config.dataset_folder,
            self.sites,
            config.single_site_config,
            max_patches_per_site=config.max_patches_per_site,
        )

        if config.validation_ratio + config.train_ratio < 1:
            (
                self.training_dataset,
                self.validation_dataset,
                _,
            ) = torch.utils.data.random_split(
                remaining_dataset,
                [
                    config.train_ratio,
                    config.validation_ratio,
                    (1 - config.train_ratio - config.validation_ratio),
                ],
            )
        else:
            (
                self.training_dataset,
                self.validation_dataset,
            ) = torch.utils.data.random_split(
                remaining_dataset, [config.train_ratio, config.validation_ratio]
            )

        # Use a cache for validation, to seep up validation steps
        if config.cache_validation_dataset:
            self.validation_dataset = CacheDataset(self.validation_dataset)

        if config.cache_testing_dataset:
            self.testing_dataset = CacheDataset(self.testing_dataset)

        logging.info(
            "%i training patches available", len(self.training_dataset)
        )  # type: ignore

        logging.info(
            "%i validation patches available",
            len(self.validation_dataset),  # type: ignore
        )

        logging.info(
            "%i testing patches available", len(self.testing_dataset)  # type: ignore
        )

    def train_dataloader(self):
        """
        Return train dataloaded (reset every time this method is called)
        """
        return torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.config.batch_size,
            drop_last=True,
            num_workers=self.config.num_workers,
            collate_fn=batch_data_collate_fn,
            shuffle=True,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Return validation data loader (never reset)
        """
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.config.testing_validation_batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=batch_data_collate_fn,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        Return test data loader (never reset)
        """
        return torch.utils.data.DataLoader(
            self.testing_dataset,
            batch_size=self.config.testing_validation_batch_size,
            drop_last=True,
            shuffle=False,
            collate_fn=batch_data_collate_fn,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
        )


class WorldStratDataModule(pl.LightningDataModule):
    """
    The datamodule based on simplified dataset
    """

    def __init__(
        self,
        dataset_index_path: str,
        min_correlation: float = 0.1,
        batch_size: int = 32,
        testing_validation_batch_size: int = 128,
        num_workers: int = 0,
        prefetch_factor: int | None = 4,
    ):
        """
        Constructor
        """
        super().__init__()

        self.batch_size = batch_size
        self.testing_validation_batch_size = testing_validation_batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.training_dataset = WorldStratDataset(
            dataset_index_path, split="train", min_correlation=min_correlation
        )
        self.validation_dataset = WorldStratDataset(
            dataset_index_path, split="val", min_correlation=min_correlation
        )
        self.testing_dataset = WorldStratDataset(
            dataset_index_path, split="test", min_correlation=min_correlation
        )
        logging.info("%i training patches available", len(self.training_dataset))
        logging.info("%i validation patches available", len(self.validation_dataset))
        logging.info("%i testing patches available", len(self.testing_dataset))

    def train_dataloader(self):
        """
        Return train dataloaded (reset every time this method is called)
        """
        return torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=batch_data_collate_fn,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Return validation data loader (never reset)
        """
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.testing_validation_batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=batch_data_collate_fn,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        Return test data loader (never reset)
        """
        return torch.utils.data.DataLoader(
            self.testing_dataset,
            batch_size=self.testing_validation_batch_size,
            drop_last=True,
            shuffle=False,
            collate_fn=batch_data_collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
        )
