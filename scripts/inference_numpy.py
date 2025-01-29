#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
Perform inference on a Sentinel2 L2A product
"""

import hashlib
import logging
import os
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
from hydra import compose, initialize_config_dir
from hydra import utils as hydra_utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from bin.predict_carn_s2 import PatchesDataset
from torchsisr import patches, training
from torchsisr.model_zoo import get_url_and_hash_by_model_name


def get_model_from_url(
    url: str = None,
    model_type: str = "carn_10_l1_lr",
    save_dir: str = "./saved/",
    hash: str = None,
):
    """
    We get a pre-trined model by url or by model name from model zoo.
    We check if file is corrupted or not by using hash, if it is available
    """
    if url is None:
        url, hash = get_url_and_hash_by_model_name(model_type)
    if not Path(save_dir).exists():
        Path(save_dir).mkdir()
    r = requests.get(url, allow_redirects=True)
    zip_ckpt_path = save_dir + model_type + ".zip"

    if r.ok:
        open(zip_ckpt_path, "wb").write(r.content)
    else:  # HTTP status code 4XX/5XX
        print(f"Download failed: status code {r.status_code}\n{r.text}")
    print(hashlib.md5(open(zip_ckpt_path, "rb").read()).hexdigest())
    # We check if the downloaded file is not corrupted
    if hash is not None:
        downloaded_file_hash = hashlib.md5(open(zip_ckpt_path, "rb").read()).hexdigest()
        assert downloaded_file_hash == hash, "Invalid zip file!"

    # We extract zipfile and then delete it
    save_dir_extracted = save_dir + model_type + "/"
    with open(zip_ckpt_path, "rb") as zip_file:
        z = zipfile.ZipFile(zip_file)
        z.extractall(save_dir_extracted)
        z.close()
    Path(zip_ckpt_path).unlink()
    ckpt_path, config_path = (
        save_dir_extracted + model_type + ".ckpt",
        save_dir_extracted + "config.yaml",
    )
    return save_dir_extracted, ckpt_path, config_path


def predict_numpy(
    image_array: np.array,
    pretrained_path: str,
    model_name: str,
    image_array_20: np.array = None,
    mask: np.array = None,
    tile_size: int = 512,
    batch_size: int = 1,
    cpu_is_used: bool = False,
) -> np.array:
    """
    We predict a SR image from a numpy array.
    Depending on the input format, different arrays are passed to the function.
    image_array has to be defined, image_array_20 is optional.

    Only one set of bands is given (10m or 20m):
        image_array.shape = [4, H, W] ; bands order: B02, B03, B04, B08
        image_array_20 = None
    Both 10m and 20m are given, both at  their original resolution:
        image_array.shape = [4, H, W] ; bands order: B02, B03, B04, B08
        image_array_20.shape = [4, H/2, W/2] ; bands order: B05, B06, B07, B8A
    Both 10m and 20m are given, but 20m bands are already upsampled to 10m:
        image_array.shape = [8, H, W] ; bands order: B02, B03, B04, B08, B05, B06, B07, B8A
        image_array_20 = None

    pretrained_path :   path to the folder with pretrained model and hydra configuration
                        note that hydra configuration file should be always called config.yaml
    model_name :        name of the pretrained model, see model_zoo file to discover
                        the available ones
    mask :              cloud and nodata mask. Has not been tested yet
    tile_size :         tile size for carn model, it better should be 256 or 512
    batch_size :        depends on tile size. 5 for 256 and 1 for 512
    cpu_is_used :       if we explicitly want to process data on CPU
    """

    # Configure logging
    logging.basicConfig(
        level="INFO",
        datefmt="%y-%m-%d %H:%M:%S",
        format="%(asctime)s :: %(levelname)s :: %(message)s",
    )

    # Find on which device to run
    dev = torch.device("cpu")
    if torch.cuda.is_available() and not cpu_is_used:
        torch.cuda.init()
        dev = torch.device("cuda")
    logging.info("Processing will happen on device %s", str(dev))
    pretrained_path = os.path.abspath(pretrained_path) + "/"

    ckpt_path = pretrained_path + model_name + ".ckpt"

    with initialize_config_dir(version_base=None, config_dir=pretrained_path):
        cfg = compose(config_name="config.yaml")
        srnet = hydra_utils.instantiate(cfg.model.network)
        metrics = hydra_utils.instantiate(cfg.metrics.list)
        losses = hydra_utils.instantiate(cfg.losses.list)
        model = training.SuperResolutionTrainingModule.load_from_checkpoint(
            ckpt_path, srnet=srnet, metrics=metrics, losses=losses, map_location=dev
        )
    model.srnet = model.srnet.to(device=dev)

    # We load different parameters from the checkpoint
    load_10m_data = cfg.datamodule.load_10m_data
    load_20m_data = cfg.datamodule.load_20m_data

    margin = model.srnet.compute_prediction_margin()

    # TODO: in the future delete if "carn" in dict(model.srnet.named_children()).keys():
    if "carn" not in dict(model.srnet.named_children()) and not (
        load_10m_data and load_20m_data
    ):
        margin = {"10": margin} if load_10m_data else {"20": margin}
    if load_10m_data and load_20m_data:  # both band sets have the same shape
        if (
            image_array.shape[0] == 8
        ):  # if we have both 10m and 20m, but 20m is already upsampled to 10m
            scale_factor_dict = {
                "10": model.srnet.carn.upsampling_factor,
                "20": model.srnet.carn.upsampling_factor,
            }
            tile_size = {"10": tile_size, "20": tile_size}
            margin["20"] = margin["10"]
        else:  # Both 10m and 20m bands have their original resolution and come
            # in separate arrays
            scale_factor_dict = {
                "10": model.srnet.carn.upsampling_factor,
                "20": model.srnet.carn_20m.upsampling_factor,
            }
            tile_size = {"10": tile_size, "20": int(tile_size / 2)}
    else:
        if "carn" in dict(model.srnet.named_children()):
            scale_factor = model.srnet.carn.upsampling_factor
        else:
            scale_factor = model.srnet.upsampling_factor
        scale_factor_dict = (
            {"10": scale_factor} if load_10m_data else {"20": scale_factor}
        )
        tile_size = {"10": tile_size} if load_10m_data else {"20": tile_size}

    # Read numpy array as two separate sets of 10m and 20m bands (if both are loaded)
    # TODO : what to do with data formats
    if (
        np.issubdtype(image_array.dtype, np.integer) or image_array.max() > 10
    ):  # Apply scaling
        image_array = image_array / 10000.0
        image_array_20 = (
            image_array_20 / 10000.0 if image_array_20 is not None else None
        )
    # Cast to float tensor
    image_array = torch.Tensor(image_array).to(dtype=torch.float32)
    image_array_20 = (
        torch.Tensor(image_array_20).to(dtype=torch.float32)
        if image_array_20 is not None
        else None
    )

    # Get input data in carn model format where each set of bands is in different dictionary key
    if len(image_array) == 8:
        input_dict = {"source_10": image_array[:4], "source_20": image_array[4:]}
    elif image_array_20 is None:
        input_dict = (
            {"source_10": image_array} if load_10m_data else {"source_20": image_array}
        )
    else:
        input_dict = {"source_10": image_array, "source_20": image_array_20}

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

    dataset = PatchesDataset(input_dict_patches_s2)
    loader = DataLoader(dataset, batch_size=batch_size)

    upsampling_factor_ = model.upsampling_factor[
        list(model.upsampling_factor.keys())[0]
    ]
    patches_shape = input_dict_patches_s2[list(input_dict_patches_s2.keys())[0]].shape
    predicted_patches = {
        "source_unstd_"
        + key[-2:]: torch.empty(
            (
                0,
                patches_shape[1],
                int(patches_shape[2] * upsampling_factor_),
                int(patches_shape[3] * upsampling_factor_),
            )
        )
        for key in input_dict_patches_s2.keys()
    }

    # cpu_count: int = 1
    # os_cpu_count = os.cpu_count()
    # if os_cpu_count is not None:
    #     cpu_count = os_cpu_count
    # torch.set_num_threads(cpu_count)
    # print("Num threads", torch.get_num_threads())
    model.srnet.eval()
    for _, b in tqdm(
        enumerate(loader), total=int(len(dataset) / batch_size), desc="Processing"
    ):
        # We make patch-wise predictions and stack the results in
        # predicted_patches dictionary with torch.cat() The predicted
        # patches are directly rescaled to int and the nan values are
        # changed to nodata=-10000

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
            predicted_patches["source_unstd_" + list(input_dict_std.keys())[0][-2:]] = (
                torch.cat(
                    (
                        predicted_patches[
                            "source_unstd_" + list(input_dict_std.keys())[0][-2:]
                        ],
                        output_network,
                    ),
                    0,
                )
            )
        else:  # if both 10m and 20m bands are inside
            output_network_list = torch.split(
                output_network, dim=1, split_size_or_sections=4
            )

            predicted_patches["source_unstd_10"] = torch.cat(
                (
                    predicted_patches["source_unstd_10"],
                    (model.unstandardize_bands(output_network_list[0], "10") * 10000)
                    .nan_to_num(-10000)
                    .to(dtype=torch.int16),
                ),
                0,
            )
            predicted_patches["source_unstd_20"] = torch.cat(
                (
                    predicted_patches["source_unstd_20"],
                    (model.unstandardize_bands(output_network_list[1], "20") * 10000)
                    .nan_to_num(-10000)
                    .to(dtype=torch.int16),
                ),
                0,
            )

    # We reintroduce the nodata patches in the predictions
    predicted = {}
    predicted.update(
        {
            "bands_"
            + key[-2:]: torch.full(
                (
                    patches_s2.shape[0],
                    patches_s2.shape[1],
                    int(patches_shape[2] * upsampling_factor_),
                    int(patches_shape[3] * upsampling_factor_),
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

    # TODO check how mask works
    # We upscale the provided mask to the final resolution
    # If mask not provided, it is a zero array
    if mask is not None:
        mask = mask.repeat(
            scale_factor_dict[list(scale_factor_dict.keys())[0]], axis=0
        ).repeat(scale_factor_dict[list(scale_factor_dict.keys())[0]], axis=1)
    else:
        mask = np.zeros_like(predicted[list(predicted.keys())[0]][0], dtype=int)

    # Apply mask
    for key, predicted_img in predicted.items():
        for b in range(predicted_img.shape[0]):
            predicted_img[b, ...][torch.tensor(mask) > 0] = -10000
        predicted[key] = predicted_img

    # Return 4- or 8-bands single predicted int numpy array
    if len(image_array) == 8 or image_array_20 is not None:
        return torch.cat((predicted["bands_10"], predicted["bands_20"]), 0).numpy()

    return predicted[list(predicted.keys())[0]].numpy()
