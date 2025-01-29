#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
Perform inference on a Sentinel2 L2A product
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from affine import Affine
from codecarbon import OfflineEmissionsTracker
from hydra import compose, initialize_config_dir
from hydra import utils as hydra_utils
from sensorsio import sentinel2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torchsisr import patches, training


class PatchesDataset(Dataset):
    """
    Small dataset helper class
    """

    def __init__(self, patches_dict):
        self.patches_dict = patches_dict

    def __len__(self) -> int:
        return len(self.patches_dict[list(self.patches_dict.keys())[0]])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.patches_dict.items()}


def reference_predict(
    input_tensor: torch.Tensor, upsampling_factor: float
) -> torch.Tensor:
    """
    Reference forward pass
    """
    return torch.nn.functional.interpolate(
        input_tensor,
        scale_factor=upsampling_factor,
        align_corners=False,
        mode="bicubic",
    )


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    out_parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Use CARN network to super-resolve a Sentinel2 L2A image",
    )

    # Logs and paths
    out_parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: INFO. Should be one of "
        "(DEBUG, INFO, WARNING, ERROR, CRITICAL))",
    )
    out_parser.add_argument(
        "--input", "-i", type=str, help="Path to input product", required=True
    )
    out_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to output folder."
        'The results will be saved inside as "bands_10.tif" or "bands_20.tif"',
        required=True,
    )
    out_parser.add_argument(
        "--checkpoint", "-cp", type=str, help="Path to model checkpoint", required=True
    )
    out_parser.add_argument(
        "--config", "-cfg", type=str, help="Path to hydra config", required=True
    )
    # Model parameters
    out_parser.add_argument(
        "--bicubic_reference", action="store_true", help="Bicubic reference prediction"
    )
    out_parser.add_argument(
        "--tile_size",
        "-t",
        type=int,
        default=512,
        help="Tile size for tile-wise processing.",
    )
    out_parser.add_argument(
        "--predict_batch_size",
        "-b",
        type=int,
        default=1,
        help="Batch size for inference.",
    )

    return out_parser


if __name__ == "__main__":
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    # Configure logging
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.loglevel}")

    logging.basicConfig(
        level=numeric_level,
        datefmt="%y-%m-%d %H:%M:%S",
        format="%(asctime)s :: %(levelname)s :: %(message)s",
    )
    logging.getLogger("codecarbon").setLevel(logging.INFO)

    # Init carbon tracker
    tracker = OfflineEmissionsTracker(
        country_iso_code="FRA", output_dir=args.output, measure_power_secs=600
    )
    # Find on which device to run
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.init()
        dev = torch.device("cuda")
    logging.info("Processing will happen on device %s", str(dev))

    # Load Sentinel2 image
    s2_ds = sentinel2.Sentinel2(args.input)

    logging.info("Sentinel2 image: %s", str(s2_ds))

    # Load CARN model
    # cp = torch.load(args.checkpoint, map_location=dev)
    # checkpoints = cp['callbacks'][
    #     "ModelCheckpoint{'monitor': 'validation/total_loss', 'mode': 'min', \
    #     'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None, \
    #     'save_on_train_epoch_end': False}"]
    # print(checkpoints)

    # best_model_path = checkpoints['best_model_path']

    # init the carbon footprint tracker
    tracker.start()

    with initialize_config_dir(version_base=None, config_dir=args.config):
        cfg = compose(config_name="config.yaml")
        srnet = hydra_utils.instantiate(cfg.model.network)
        metrics = hydra_utils.instantiate(cfg.metrics.list)
        losses = hydra_utils.instantiate(cfg.losses.list)
        model = training.SuperResolutionTrainingModule.load_from_checkpoint(
            args.checkpoint,
            srnet=srnet,
            metrics=metrics,
            losses=losses,
            map_location=dev,
        )
    model.srnet = model.srnet.to(device=dev)

    # We load different parameters from the checkpoint
    load_10m_data = cfg.datamodule.load_10m_data
    load_20m_data = cfg.datamodule.load_20m_data
    # TODO: in the future delete if "carn" in dict(model.srnet.named_children()):
    margin = model.srnet.compute_prediction_margin()
    if "carn" not in dict(model.srnet.named_children()) and not (
        load_10m_data and load_20m_data
    ):
        margin = {"10": margin} if load_10m_data else {"20": margin}

    if load_10m_data and load_20m_data:
        scale_factor_dict = {
            "10": model.srnet.carn.upsampling_factor,
            "20": model.srnet.carn_20m.upsampling_factor,
        }
        tile_size = {"10": args.tile_size, "20": int(args.tile_size / 2)}
    else:
        if "carn" in dict(model.srnet.named_children()):
            scale_factor = model.srnet.carn.upsampling_factor
        else:
            scale_factor = model.srnet.upsampling_factor
        scale_factor_dict = (
            {"10": scale_factor} if load_10m_data else {"20": scale_factor}
        )
        tile_size = {"10": args.tile_size} if load_10m_data else {"20": args.tile_size}

    predict_batch_size = args.predict_batch_size

    # Read image as two separate sets of 10m and 20m bands (if both are loaded)
    input_dict = {}
    if load_10m_data:
        bands, masks, _, xcoords, ycoords, crs = s2_ds.read_as_numpy(
            sentinel2.Sentinel2.GROUP_10M, resolution=10
        )
        bands_tensor = torch.tensor(bands)
        if bands_tensor.dtype == torch.int16:
            bands_tensor = (bands_tensor / 10000.0).to(
                dtype=torch.float32
            )  # Apply scaling
        # bands = bands[:, -4000:, -4000:]
        input_dict.update({"source_10": bands_tensor})
    if load_20m_data:
        bands_20, masks_20, _, xcoords, ycoords, crs = s2_ds.read_as_numpy(
            sentinel2.Sentinel2.GROUP_20M[:4], resolution=20
        )
        bands_20_tensor = torch.tensor(bands_20)
        if bands_20_tensor.dtype == torch.int16:
            bands_20_tensor = (bands_20_tensor / 10000.0).to(
                dtype=torch.float32
            )  # Apply scaling
        # bands = bands[:, -2000:, -2000:]
        input_dict.update({"source_20": bands_20_tensor})

    # Split bands into patches
    input_dict_patches_s2_all = {
        "source_"
        + key[-2:]: patches.patchify(
            value, patch_size=tile_size[key[-2:]], margin=margin[key[-2:]]
        )
        for key, value in input_dict.items()
    }

    row_nb_patches, col_nb_patches = input_dict_patches_s2_all[
        list(input_dict_patches_s2_all.keys())[0]
    ].shape[:2]

    input_dict_patches_s2_all = {
        key: patches.flatten2d(s2_patches)
        for key, s2_patches in input_dict_patches_s2_all.items()
    }

    # We only choose patches with valid data inside
    valid_patches_ind = np.where(
        [
            0 in torch.isnan(patch)
            for patch in input_dict_patches_s2_all[
                list(input_dict_patches_s2_all.keys())[0]
            ]
        ]
    )

    input_dict_patches_s2 = {
        key: s2_patches[valid_patches_ind]
        for key, s2_patches in input_dict_patches_s2_all.items()
    }
    # non_valid_patches_ind = np.where(~nodata_patch)

    nb_patches = input_dict_patches_s2[list(input_dict_patches_s2.keys())[0]].shape[0]

    dataset = PatchesDataset(input_dict_patches_s2)
    loader = DataLoader(dataset, batch_size=predict_batch_size)

    predicted_patches = {
        "source_unstd_"
        + key[-2:]: torch.empty(
            (
                0,
                patch_s2.shape[1],
                int(patch_s2.shape[2] * model.upsampling_factor[key[-2:]]),
                int(patch_s2.shape[3] * model.upsampling_factor[key[-2:]]),
            )
        )
        for key, patch_s2 in input_dict_patches_s2.items()
    }

    # TODO: use an argparse option instead
    cpu_count: int = 1
    os_cpu_count = os.cpu_count()
    if os_cpu_count is not None:
        cpu_count = os_cpu_count
    torch.set_num_threads(cpu_count)
    # torch.set_num_threads(8)
    print("Num threads", torch.get_num_threads())
    model.srnet.eval()
    # with torch.no_grad(): # already in .predict
    for _, b in tqdm(
        enumerate(loader),
        total=int(len(dataset) / predict_batch_size),
        desc="Processing",
    ):
        # We make patch-wise predictions and stack the results in
        # predicted_patches dictionary with torch.cat() The predicted
        # patches are directly rescaled to int and the nan values are
        # changed to nodata=-10000
        if args.bicubic_reference:
            # we compute a reference SR image by using simple bicubic
            # upsampling with function reference_predict()
            predicted_patches.update(
                {
                    "source_unstd_"
                    + key[-2:]: torch.cat(
                        (
                            predicted_patches["source_unstd_" + key[-2:]],
                            (
                                reference_predict(
                                    batch.to(device=dev),
                                    model.upsampling_factor[key[-2:]],
                                ).cpu()
                                * 10000
                            )
                            .nan_to_num(-10000)
                            .to(dtype=torch.int16),
                        ),
                        0,
                    )
                    for key, batch in b.items()
                }
            )
        else:
            # we compute a SR image by using pretrained CARN model
            input_dict_std = {
                "source_std_"
                + key[-2:]: model.standardize_bands(batch, key[-2:]).to(device=dev)
                for key, batch in b.items()
            }

            output_network = model.srnet.predict(input_dict_std).cpu()

            if len(b.keys()) == 1:  # if only one set of bands is used
                output_network = (
                    (
                        model.unstandardize_bands(
                            output_network, list(input_dict_std.keys())[0][-2:]
                        )
                        * 10000
                    )
                    .nan_to_num(-10000)
                    .to(dtype=torch.int16)
                )
                predicted_patches[
                    "source_unstd_" + list(input_dict_std.keys())[0][-2:]
                ] = torch.cat(
                    (
                        predicted_patches[
                            "source_unstd_" + list(input_dict_std.keys())[0][-2:]
                        ],
                        output_network,
                    ),
                    0,
                )
            else:  # if both 10m and 20m bands are inside
                output_network_list = torch.split(
                    output_network, dim=1, split_size_or_sections=4
                )

                predicted_patches["source_unstd_10"] = torch.cat(
                    (
                        predicted_patches["source_unstd_10"],
                        (
                            model.unstandardize_bands(output_network_list[0], "10")
                            * 10000
                        )
                        .nan_to_num(-10000)
                        .to(dtype=torch.int16),
                    ),
                    0,
                )
                predicted_patches["source_unstd_20"] = torch.cat(
                    (
                        predicted_patches["source_unstd_20"],
                        (
                            model.unstandardize_bands(output_network_list[1], "20")
                            * 10000
                        )
                        .nan_to_num(-10000)
                        .to(dtype=torch.int16),
                    ),
                    0,
                )

    predicted = {}

    # We reintroduce the nodata patches in the predictions
    predicted.update(
        {
            "bands_"
            + key[-2:]: torch.full(
                (
                    patches_s2.shape[0],
                    patches_s2.shape[1],
                    int(patches_s2.shape[2] * model.upsampling_factor[key[-2:]]),
                    int(patches_s2.shape[3] * model.upsampling_factor[key[-2:]]),
                ),
                -10000,
                dtype=predicted_patches[list(predicted_patches.keys())[0]].dtype,
            )
            for key, patches_s2 in input_dict_patches_s2_all.items()
        }
    )

    for key, predicted_patch in predicted_patches.items():
        new_predicted = predicted["bands_" + key[-2:]]
        new_predicted[valid_patches_ind] = predicted_patch
        predicted_patches.update({key: new_predicted})

    # We convert patches back to image
    predicted.update(
        {
            "bands_"
            + key[-2:]: patches.unpatchify(
                patches.unflatten2d(predicted_patch, row_nb_patches, col_nb_patches),
                margin=int(
                    scale_factor_dict[list(scale_factor_dict.keys())[0]]
                    * margin[list(margin.keys())[0]]
                ),
            )
            for key, predicted_patch in predicted_patches.items()
            if "source_unstd_" in key
        }
    )

    # We crop back to original shape
    predicted.update(
        {
            key: image[
                :,
                : int(
                    scale_factor_dict[list(scale_factor_dict.keys())[0]]
                    * input_dict[list(input_dict.keys())[0]].shape[1]
                ),
                : int(
                    scale_factor_dict[list(scale_factor_dict.keys())[0]]
                    * input_dict[list(input_dict.keys())[0]].shape[2]
                ),
            ]
            for key, image in predicted.items()
        }
    )

    # We get the mask and apply it band-wise
    # Read edge mask at 5m
    _, edge_mask_5m, _, _, _, _str = s2_ds.read_as_numpy(
        bands=[sentinel2.Sentinel2.B2], masks=[sentinel2.Sentinel2.EDG], resolution=5
    )

    # Drop first dimension
    edge_mask_5m = edge_mask_5m[0, ...]

    # Apply mask
    for key, predicted_img in predicted.items():
        for b in range(predicted_img.shape[0]):
            predicted_img[b, ...][torch.tensor(edge_mask_5m) > 0] = -10000
        predicted[key] = predicted_img

    # Write SR images. Each set of bands is written separately in a
    # file called "bands_10.tif" or "bands_20.tif"
    geotransform = (s2_ds.bounds[0], 5.0, 0.0, s2_ds.bounds[3], 0.0, -5.0)
    transform = Affine.from_gdal(*geotransform)

    profile = {
        "driver": "GTiff",
        "height": predicted[list(predicted.keys())[0]].shape[1],
        "width": predicted[list(predicted.keys())[0]].shape[2],
        "count": predicted[list(predicted.keys())[0]].shape[0],
        "dtype": np.int16,
        "crs": s2_ds.crs,
        "transform": transform,
        "nodata": -10000,
        "tiled": True,
    }

    for key, image in predicted.items():
        resolution = key[:-2]
        with rio.open(args.output + key + ".tif", "w", **profile) as ds:
            for band in range(image.shape[0]):
                ds.write(image[band, ...].detach().numpy().astype(np.int16), band + 1)

    tracker.stop()

    # We compute consumption per pixel
    to_keep = ["timestamp", "project_name", "run_id"]
    absolute_values = [
        "duration",
        "emissions",
        "emissions_rate",
        "cpu_power",
        "gpu_power",
        "ram_power",
        "cpu_energy",
        "gpu_energy",
        "ram_energy",
        "energy_consumed",
    ]
    relative_values = [v + "/pix" for v in absolute_values]
    pd_emissions = pd.read_csv(args.output + "emissions.csv", header=0)
    last_emission = pd_emissions[np.concatenate((to_keep, absolute_values), 0)].iloc[
        [-1], :
    ]
    total_pixels = (
        predicted[list(predicted.keys())[0]].shape[1]
        * predicted[list(predicted.keys())[0]].shape[2]
    )
    for name in absolute_values:
        last_emission[name + "/100000pix"] = last_emission[name] / total_pixels * 10000
    if os.path.isfile(args.output + "emissions_per_10000_pix.csv"):
        last_emission.to_csv(
            args.output + "emissions_per_10000_pix.csv",
            mode="a",
            index=False,
            header=False,
        )
    else:
        last_emission.to_csv(
            args.output + "emissions_per_10000_pix.csv",
            mode="a",
            index=False,
            header=True,
        )
