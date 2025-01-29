#!/usr/bin/env python

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
A tool to analyse the worldstrat dataset
"""
import argparse
import os

import numpy as np
import pandas as pd
import rasterio as rio
import torch
import yaml
from affine import Affine
from tqdm import tqdm

from torchsisr.dataset import generic_downscale


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="Analyse of WorldStrat dataset"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Directory of WorldStrat",
    )

    parser.add_argument("--max_delta", type=int, default=5, help="Maximum time delta")
    parser.add_argument(
        "--max_nb_samples",
        type=int,
        required=False,
        default=None,
        help="Maximum nb samples to extract",
    )
    parser.add_argument("--output", type=str, required=True, help="Output folder")
    return parser


def main():
    """
    Main method
    """
    # Parser arguments
    args = get_parser().parse_args()

    available_patches: list[str] = []

    csv_index = pd.read_csv(
        os.path.join(args.input, "stratified_train_val_test_split.csv")
    ).set_index("tile")

    for root, _, files in os.walk(os.path.join(args.input, "lr_dataset_l2a/")):
        if (
            args.max_nb_samples is not None
            and len(available_patches) > args.max_nb_samples
        ):
            break
        min_delta = args.max_delta
        best_patch = ""
        for f in files:
            if f.endswith(".metadata"):
                with open(os.path.join(root, f)) as fp:
                    metadata = yaml.safe_load(fp)
                    if "delta" in metadata and abs(int(metadata["delta"])) < min_delta:
                        best_patch = os.path.join(root, f)
                        min_delta = abs(int(metadata["delta"]))
        if min_delta < args.max_delta:
            available_patches.append(best_patch)

    # Restrict index to selected patches
    hr_imgs: list[str] = []
    lr_imgs: list[str] = []
    corrs: list[float] = []

    # Ensure patches are in the index
    available_patches = [
        p for p in available_patches if p.rsplit("/")[-3] in csv_index.index
    ]

    csv_index = csv_index.loc[[f.rsplit("/")[-3] for f in available_patches]]
    csv_index = csv_index[~csv_index.index.duplicated(keep="first")]
    print(csv_index)
    print(len(available_patches), len(csv_index))
    for p in tqdm(
        available_patches, total=len(available_patches), desc="Extracting patches"
    ):
        lr_image_path = p[:-9] + "-L2A_data.tiff"
        image_id = lr_image_path.rsplit("/")[-3]
        hr_image_path = os.path.join(
            args.input, "hr_dataset", image_id, image_id + "_ps.tiff"
        )
        with rio.open(lr_image_path, "r") as rio_ds:
            lr_tensor = rio_ds.read()
        with rio.open(hr_image_path, "r") as rio_ds:
            hr_tensor = rio_ds.read()

        hr_patch_size = 512
        lr_patch_size = 128

        # TODO: Make the resolution factor a parameter
        hr_tensor = generic_downscale(
            torch.tensor(hr_tensor.astype(float))[None, ...], factor=2.5 / 1.5, mtf=0.1
        ).to(dtype=torch.int16)[0, ...]

        hr_to_lr_tensor = (
            generic_downscale(hr_tensor.to(dtype=float)[None, ...], factor=4.0, mtf=0.1)
            .to(dtype=torch.int16)[0, ...]
            .numpy()
        )

        lr_tensor = lr_tensor[
            [1, 2, 3, 7], 10 : 10 + lr_patch_size, 10 : 10 + lr_patch_size
        ]

        hr_tensor = hr_tensor[
            [2, 1, 0, 3], 40 : 40 + hr_patch_size, 40 : 40 + hr_patch_size
        ]

        hr_to_lr_tensor = hr_to_lr_tensor[
            [2, 1, 0, 3], 10 : 10 + lr_patch_size, 10 : 10 + lr_patch_size
        ]

        corrs.append(
            np.abs(
                np.corrcoef(
                    lr_tensor.mean(axis=0).ravel(), hr_to_lr_tensor.mean(axis=0).ravel()
                )[0, 1]
            )
        )

        geotransform = (0, 10.0, 0.0, 0, 0.0, -10.0)
        transform = Affine.from_gdal(*geotransform)

        profile = {
            "driver": "GTiff",
            "height": lr_tensor.shape[1],
            "width": lr_tensor.shape[2],
            "count": lr_tensor.shape[0],
            "dtype": np.int16,
            "transform": transform,
        }
        lr_tensor = 10000 * lr_tensor
        lr_output_path = os.path.join(args.output, image_id + "_lr.tif")
        lr_imgs.append(os.path.basename(lr_output_path))
        with rio.open(lr_output_path, "w", **profile) as rio_dataset:
            for band in range(lr_tensor.shape[0]):
                rio_dataset.write(lr_tensor[band, ...].astype(np.int16), band + 1)

        geotransform = (0, 2.5, 0.0, 0, 0.0, -2.5)
        transform = Affine.from_gdal(*geotransform)

        profile = {
            "driver": "GTiff",
            "height": hr_tensor.shape[1],
            "width": hr_tensor.shape[2],
            "count": hr_tensor.shape[0],
            "dtype": np.int16,
            "transform": transform,
        }

        hr_output_path = os.path.join(args.output, image_id + "_hr.tif")
        hr_imgs.append(os.path.basename(hr_output_path))

        with rio.open(hr_output_path, "w", **profile) as rio_dataset:
            for band in range(hr_tensor.shape[0]):
                rio_dataset.write(
                    hr_tensor[band, ...].numpy().astype(np.int16), band + 1
                )

    csv_index["hr_img"] = hr_imgs
    csv_index["lr_img"] = lr_imgs
    csv_index["correlation"] = corrs
    csv_index.to_csv(os.path.join(args.output, "index.csv"), sep="\t")


if __name__ == "__main__":
    main()
