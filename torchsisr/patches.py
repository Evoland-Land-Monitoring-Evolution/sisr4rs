#!/usr/bin/env python
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
Helper patches manipulation methods
"""


import numpy as np
import torch


def patchify(
    data: torch.Tensor,
    patch_size: int = 32,
    margin: int = 0,
    padding_mode: str = "constant",
    padding_value: int = 0,
    spatial_dim1: int = 1,
    spatial_dim2: int = 2,
) -> torch.Tensor:
    """
    Create a patch view on an image Tensor
    data: Tensor of shape [C,W,H] (C: number of channels,
                                   W: image width, H: image height)
    :param patch_size: Size of the square patch
    :param margin: Overlap of patches on each side
    :param padding_mode: Mode for padding on left/bottom end
    :param padding_value: Value of padding if padding_mode is 'constant'
    return: Tensor of shape [PX, PY, C, PW, PH]
            (PX: patch x idx, PY: patch y idx, C: number of channels,
             PW: patch width, PH: patch height) Lastes patch might be padded
    """
    # First, move spatial dims to the end
    data = data.transpose(spatial_dim1, -2).transpose(spatial_dim2, -1)

    padding_left = margin
    padding_right = (
        margin + int(np.ceil(data.shape[-2] / patch_size) * patch_size) - data.shape[-2]
    )
    padding_top = margin
    padding_bottom = (
        margin + int(np.ceil(data.shape[-1] / patch_size) * patch_size) - data.shape[-1]
    )

    return (
        (
            torch.nn.functional.pad(
                data[None, ...],
                [padding_top, padding_bottom, padding_left, padding_right],
                mode=padding_mode,
                value=padding_value,
            )[0, ...]
            .unfold(-2, patch_size + 2 * margin, patch_size)
            .unfold(-2, patch_size + 2 * margin, patch_size)
        )
        .transpose(-4, 0)
        .transpose(-3, 1)
    )  # Put spatial dimensions back in place


def unpatchify(data: torch.Tensor, margin=0) -> torch.Tensor:
    """
    :param data: Tensor of shape [PX,PY, C, PW, PH]
    :param margin:
    :returns: Tensor of shape [C,W,H] with W=PX*(PW-2*margin) and H=PY*(PH-2*margin)

    Sizes might be greater than original images because of
    padding. Consider cropping to original dimensions
    """
    data = crop(data, margin)
    final_shape = (
        data.shape[2],
        data.shape[0] * data.shape[3],
        data.shape[1] * data.shape[4],
    )
    out = torch.zeros(final_shape)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            out[
                :,
                r * data.shape[3] : (r + 1) * data.shape[3],
                c * data.shape[4] : (c + 1) * data.shape[4],
            ] = data[r, c, ...]
    return out


def flatten2d(
    data: torch.Tensor, spatial_dim1: int = 0, spatial_dim2: int = 1
) -> torch.Tensor:
    """
    Convert a Tensor with 2d dimensions as spatial_dim1, spatial_dim2
    to a flatten Tensor collapsing those dimensions
    data: Tensor of shape [PX, PY, C, PW, PH] or [W, H, C]
          (PX: patch x idx, PY: patch y idx, C: number of channels,
           PW: patch width, PH: patch height, W: image width, H: image height)
    return: Tensor of shape [PX*PY, C, PW, PH] or [W*H, C]
    """
    return data.flatten(spatial_dim1, spatial_dim2)


def unflatten2d(data: torch.Tensor, h: int, w: int, batch_dim: int = 0) -> torch.Tensor:
    """
    Unflatten batch_dim dimension of a Tensor to 2 dimensions
    data: Tensor of shape [N, C, PW, PH] or [N, C]
    w: Width (first dimension)
    h: Height (second dimension)
    return: Tensor of shape [W, H, C, PW, PH] or [W, H, C]
    """
    return data.unflatten(batch_dim, (h, w))


def flatten_patches(data: torch.Tensor) -> torch.Tensor:
    """
    Flatten A 4d image patches tensor to a 2d tensor
    data: Tensor of shape [N, C, PW, PH]
    return: Tensor of shape [N*PW*PH, C]
    """
    return data.transpose(1, 2).transpose(2, 3).flatten(0, 2)


def unflatten_patches(data: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Unflatten first dimension to 2 extra patch dimensions at the end
    data: Tensor of shape [N*patch_size, C]
    return: Tensor of shape [N, C, patch_size, patch_size]
    """
    nb_samples = data.shape[0] // (patch_size * patch_size)
    return (
        data.unflatten(0, (nb_samples, patch_size, patch_size))
        .transpose(3, 2)
        .transpose(2, 1)
    )


def crop(data: torch.Tensor, crop_length: int = 0) -> torch.Tensor:
    """
    Crop patches
    """
    if crop_length == 0:
        return data

    return data[..., crop_length:-crop_length, crop_length:-crop_length]


def standardize(
    data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """
    Apply standardization from mean/std estimates
    :param data: input tensor of shape [nb_patches, nb_channels, patch_size_x, patch_size_y]
    :param mean: mean tensor of shape [nb_channels]
    :param std: std tensor of shape [nb_channels]
    :return: standardized tensor of shape [nb_patches, nb_channels, patch_size_x, patch_size_y]
    """
    return (data - mean[None, :, None, None]) / std[None, :, None, None]


def unstandardize(
    data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """
    Apply unstandardization from mean/std estimates
    :param data: input tensor of shape [nb_patches, nb_channels, patch_size_x, patch_size_y]
    :param mean: mean tensor of shape [nb_channels]
    :param std: std tensor of shape [nb_channels]

    :return: unstandardized tensor of shape [nb_patches, nb_channels,
        patch_size_x, patch_size_y]
    """
    return data * std[None, :, None, None] + mean[None, :, None, None]
