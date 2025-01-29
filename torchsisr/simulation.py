# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Code for the simulation of patches for the metrics study
"""
import numpy as np
import torch
from einops import repeat

from torchsisr.dataset import generate_psf_kernel


def simulate(
    data: torch.Tensor,
    mtf: float | None = 0.3,
    spectral_distorsion: float | None = 0.1,
    spatial_distorsion: float | None = 0.1,
    periodic_pattern: float | None = 0.1,
    noise_std: float | None = 0.1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Simulate data from Venµs patches

    """
    if spatial_distorsion is not None:
        # Apply spatial deformation
        x = (
            torch.linspace(0.5, data.shape[3] - 1 + 0.5, data.shape[3], device=device)
            - 0.5 * data.shape[3]
        )
        y = (
            torch.linspace(0.5, data.shape[2] - 1 + 0.5, data.shape[2], device=device)
            - 0.5 * data.shape[2]
        )
        x += spatial_distorsion / np.sqrt(2)
        y += spatial_distorsion / np.sqrt(2)
        x /= 0.5 * data.shape[3]
        y /= 0.5 * data.shape[2]
        deformation_field = torch.stack(torch.meshgrid(x, y), dim=-1)
        deformation_field = repeat(
            deformation_field, "h w d -> b w h d", b=data.shape[0]
        )

        data = torch.nn.functional.grid_sample(
            data, deformation_field, mode="bicubic", align_corners=False
        )

    # Apply spectral distorsion
    if spectral_distorsion is not None:
        data = 0.1 + (data - 0.1) * (1.0 - spectral_distorsion)

    if mtf is not None:
        # Apply blur
        blur_kernel = torch.tensor(
            generate_psf_kernel(1.0, 1.0, mtf_fc=mtf, half_kernel_width=3),
            device=device,
        )

        # pylint: disable=not-callable
        data = torch.nn.functional.conv2d(
            data,
            blur_kernel[None, None, :, :].expand(data.shape[1], -1, -1, -1),
            groups=data.shape[1],
            padding="same",
        )

    if noise_std is not None:
        noise = torch.normal(0.0, torch.full_like(data, noise_std))
        data += noise

    if periodic_pattern is not None:
        period = 4
        small_pattern = torch.tensor(
            [
                [-0.2958, 0.0351, -0.4362, 0.2725],
                [0.2069, 0.3313, 0.2105, -0.0692],
                [0.3401, -0.3956, 0.2525, -0.0705],
                [0.4418, 0.2502, -0.4821, -0.3385],
            ],
            device=data.device,
        )
        pattern = torch.cat(
            [small_pattern for i in range(data.shape[-2] // period)], dim=0
        )
        pattern = torch.cat([pattern for i in range(data.shape[-1] // period)], dim=1)
        data += periodic_pattern * pattern

    return data
