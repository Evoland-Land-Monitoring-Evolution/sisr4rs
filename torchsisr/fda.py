#!/usr/bin/env python
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales

"""
Contains Frequency Domain Analysis related functions
"""

import numpy as np
import torch
from matplotlib import pyplot as plt


def compute_fft_profile(
    data: torch.Tensor, s: int = 512
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute fft profile for given tensor, using circles of increasing frequencies
    """
    out_fft = torch.stack(
        [
            torch.abs(
                # pylint: disable=not-callable
                torch.fft.fftshift(
                    # pylint: disable=not-callable
                    torch.fft.fft2(
                        data[:, i, ...].to(dtype=torch.float32),
                        norm="backward",
                        s=[s, s],
                    )
                )
            )
            for i in range(data.shape[1])
        ],
        dim=1,
    ).mean(dim=0, keepdim=True)

    # pylint: disable=not-callable
    freqs = torch.fft.fftshift(torch.fft.fftfreq(s, d=float(data.shape[-1]) / s))

    half_freqs = freqs[freqs.shape[0] // 2 :]

    freqs_x, freqs_y = torch.meshgrid(freqs, freqs)

    freq_dist = torch.sqrt(freqs_x**2 + freqs_y**2)

    fft_prof = torch.stack(
        [
            out_fft[:, :, torch.logical_and(f1 < freq_dist, freq_dist < f2)].mean(dim=2)
            for f1, f2 in zip(half_freqs[:-1:2], half_freqs[1::2])
        ],
        dim=1,
    )
    freq_values = torch.tensor(
        [(f1 + f2) / 2 for f1, f2 in zip(half_freqs[:-1:2], half_freqs[1::2])]
    )

    return out_fft, freq_values, fft_prof


def plot_fft_profile(profiles: np.ndarray, freqs: np.ndarray, output_pdf: str):
    """
    plot fft data
    """

    labels_10m = ["B2", "B3", "B4", "B8"]
    colors_10m = ["blue", "green", "red", "brown"]

    labels_20m = ["B5", "B6", "B7", "B8A"]
    colors_20m = ["orange", "goldenrod", "gold", "peru"]

    nrows = 1
    height = 5
    if profiles.shape[-1] > 4:
        nrows = 2
        height = 10
    fig, ax = plt.subplots(nrows=nrows, ncols=4, figsize=(20, height), squeeze=False)
    for i in range(4):
        ax[0, i].plot(
            freqs,
            10 * (np.log10(profiles[1, :, i]) - np.log10(profiles[1, 1, i])),
            label=labels_10m[i] + " (Venµs)",
            color=colors_10m[i],
        )

        ax[0, i].plot(
            freqs,
            10 * (np.log10(profiles[0, :, i]) - np.log10(profiles[0, 1, i])),
            label=labels_10m[i] + " (Sentinel-2)",
            color=colors_10m[i],
            ls="--",
        )

        ax[0, i].plot(
            freqs,
            10 * (np.log10(profiles[2, :, i]) - np.log10(profiles[2, 1, i])),
            label=labels_10m[i] + " (S2 SISR)",
            color=colors_10m[i],
            ls="dotted",
        )
    if profiles.shape[-1] > 4:
        for i in range(4):
            ax[1, i].plot(
                freqs,
                10
                * (np.log10(profiles[1, :, 4 + i]) - np.log10(profiles[1, 1, 4 + i])),
                label=labels_20m[i] + " (Venµs)",
                color=colors_20m[i],
            )
            ax[1, i].plot(
                freqs,
                10
                * (np.log10(profiles[0, :, 4 + i]) - np.log10(profiles[0, 1, 4 + i])),
                label=labels_20m[i] + " (Sentinel-2)",
                color=colors_20m[i],
                ls="--",
            )
            ax[1, i].plot(
                freqs,
                10
                * (np.log10(profiles[2, :, 4 + i]) - np.log10(profiles[2, 1, 4 + i])),
                label=labels_20m[i] + " (S2 SISR)",
                color=colors_20m[i],
                ls="dotted",
            )

    for x in ax.ravel():
        x.grid(True)
        x.legend()
        #        x.set_ylim([-35, 0])
        x.set_ylabel("Signal attenuation (dB)")
        x.set_xlabel("Spatial freq. (1/px)")

    fig.savefig(output_pdf)


def compute_frr(
    predicted_logprof: torch.Tensor,
    target_logprof: torch.Tensor,
    input_logprof: torch.Tensor,
    fmin: float = 0.0,
    fmax: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the Frequency Restoration Rate metric
    """

    idx_min = int(fmin * target_logprof.shape[0])
    idx_max = int(fmax * (target_logprof.shape[0] - 1))
    if idx_min == 0:
        idx_min = 1

    pfr = (
        target_logprof[idx_min:idx_max].sum()
        - torch.minimum(
            target_logprof[idx_min:idx_max], input_logprof[idx_min:idx_max]
        ).sum()
    )

    prr = torch.abs(
        pfr
        / torch.minimum(
            target_logprof[idx_min:idx_max], input_logprof[idx_min:idx_max]
        ).sum()
    )

    afr = (
        torch.maximum(
            torch.minimum(
                predicted_logprof[idx_min:idx_max],
                target_logprof[idx_min:idx_max],
            ),
            torch.minimum(
                input_logprof[idx_min:idx_max],
                target_logprof[idx_min:idx_max],
            ),
        ).sum()
        - torch.minimum(
            target_logprof[idx_min:idx_max], input_logprof[idx_min:idx_max]
        ).sum()
    )

    arr = torch.abs(
        afr
        / torch.minimum(
            target_logprof[idx_min:idx_max], input_logprof[idx_min:idx_max]
        ).sum()
    )

    return 100 * arr / prr, 100 * arr, 100 * prr


def compute_fro_fru(
    predicted_logprof: torch.Tensor,
    target_logprof: torch.Tensor,
    input_logprof: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the Frequency Restoration Overshoot and Frequency
    Restoration Undershoot metrics
    """
    # Over-restoration
    fro1 = target_logprof[1:].sum() - (
        torch.maximum(predicted_logprof[1:], target_logprof[1:]).sum()
    )
    fro2 = target_logprof[1:].sum()

    fru1 = (
        input_logprof[1:] - torch.minimum(predicted_logprof[1:], input_logprof[1:])
    ).sum()
    fru2 = input_logprof[1:].sum()

    return 100 * fro1 / fro2, 100 * fru1 / fru2


def compute_fda_noise(
    predicted_logprof: torch.Tensor,
    target_logprof: torch.Tensor,
    input_logprof: torch.Tensor,
    low_point: float = 0.6,
    mid_point: float = 0.8,
) -> torch.Tensor:
    """
    Estimate level of noise from Frequency Domain Analysis (EXPERIMENTAL)
    """
    frr_low_mid = compute_frr(
        predicted_logprof, target_logprof, input_logprof, low_point, mid_point
    )[0]
    frr_mid_max = compute_frr(
        predicted_logprof, target_logprof, input_logprof, mid_point, 1.0
    )[0]
    return torch.clip(frr_mid_max - frr_low_mid, min=0.0)
