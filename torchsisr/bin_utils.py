# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
A tool to generate samples for training of super-resolution algorithms
"""

import numpy as np
import rasterio as rio
import torch
from affine import Affine


def write_image(
    img: torch.Tensor, img_path: str, ncols: int = 8, nrows: int = 4, res: float = 1.0
):
    """
    Write image mosaic utility function
    """
    img = img.unflatten(0, (ncols, nrows))

    img = torch.cat([img[i, ...] for i in range(ncols)], dim=-2)
    img_np = torch.cat([img[i, ...] for i in range(nrows)], dim=-1).numpy()

    geotransform = (0, res, 0.0, 0, 0.0, -res)
    transform = Affine.from_gdal(*geotransform)

    profile = {
        "driver": "GTiff",
        "height": img_np.shape[1],
        "width": img_np.shape[2],
        "count": img_np.shape[0],
        "dtype": np.float32,
        "transform": transform,
    }

    with rio.open(img_path, "w", **profile) as rio_dataset:
        for band in range(img_np.shape[0]):
            rio_dataset.write(img_np[band, ...], band + 1)
