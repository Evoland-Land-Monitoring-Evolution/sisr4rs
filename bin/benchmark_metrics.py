#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
A tool to generate warped images
"""

import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
from einops import repeat
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from piq import (  # type: ignore
    CLIPIQA,
    DISTS,
    LPIPS,
    PieAPP,
    brisque,
    dss,
    fsim,
    gmsd,
    haarpsi,
    information_weighted_ssim,
    mdsi,
    multi_scale_gmsd,
    multi_scale_ssim,
    psnr,
    srsim,
    ssim,
    total_variation,
    vif_p,
    vsi,
)
from tqdm import tqdm

from torchsisr import dataset, fda, simulation

lpips = LPIPS()
pieapp = PieAPP()
dists = DISTS()
clipiqa = CLIPIQA().cuda()


def compute_frr(
    pred: torch.Tensor,
    ref: torch.Tensor,
    bicubic_input: torch.Tensor,
    fmin: float = 0.0,
    fmax: float = 1.0,
) -> torch.Tensor:
    """
    Compute Frequency Restoration Rate
    """

    _, _, target_prof = fda.compute_fft_profile(ref)
    _, _, pred_prof = fda.compute_fft_profile(pred)
    _, _, input_prof = fda.compute_fft_profile(bicubic_input)

    idx_min = int(fmin * target_prof.shape[-2])
    idx_max = int(fmax * (target_prof.shape[-2] - 1))
    if idx_min == 0:
        idx_min = 1

    target_logprof = 10 * torch.log10(target_prof) - 10 * torch.log10(
        target_prof[:, 1:2, ...]
    )
    input_logprof = 10 * torch.log10(input_prof) - 10 * torch.log10(
        input_prof[:, 1:2, ...]
    )
    predicted_logprof = 10 * torch.log10(pred_prof) - 10 * torch.log10(
        pred_prof[:, 1:2, ...]
    )

    pfr = (
        target_logprof[:, idx_min:idx_max, 0].sum()
        - torch.minimum(
            target_logprof[:, idx_min:idx_max, 0], input_logprof[:, idx_min:idx_max, 0]
        ).sum()
    )
    afr = (
        torch.maximum(
            torch.minimum(
                predicted_logprof[:, idx_min:idx_max, 0],
                target_logprof[:, idx_min:idx_max, 0],
            ),
            torch.minimum(
                input_logprof[:, idx_min:idx_max, 0],
                target_logprof[:, idx_min:idx_max, 0],
            ),
        ).sum()
        - torch.minimum(
            target_logprof[:, idx_min:idx_max, 0], input_logprof[:, idx_min:idx_max, 0]
        ).sum()
    )
    return 100 * afr / pfr


def compute_metrics(pred: torch.Tensor, ref: torch.Tensor, bicubic_input: torch.Tensor):
    """
    Compute a set of metrics
    """
    pred = torch.clip(pred, 0.0, 1.0)
    ref = torch.clip(ref, 0.0, 1.0)
    pred3 = repeat(pred, "b 1 w h -> b 3 w h")
    ref3 = repeat(ref, "b 1 w h -> b 3 w h")

    return torch.stack(
        [
            psnr(pred, ref, data_range=1.0),
            ssim(pred, ref, data_range=1.0),
            multi_scale_ssim(pred, ref, data_range=1.0),
            information_weighted_ssim(pred, ref, data_range=1.0),
            gmsd(pred, ref, data_range=1.0),
            multi_scale_gmsd(pred, ref, data_range=1.0),
            brisque(pred),
            brisque(ref) - brisque(pred),
            total_variation(ref) - total_variation(pred),
            dss(pred, ref, data_range=1.0),
            haarpsi(pred, ref, data_range=1.0),
            torch.nn.functional.l1_loss(pred, ref),
            torch.nn.functional.mse_loss(pred, ref),
            vif_p(pred, ref),
            fsim(pred3, ref3),
            srsim(pred, ref),
            mdsi(pred, ref),
            lpips(pred3, ref3),
            pieapp(pred3, ref3),
            dists(pred3, ref3),
            vsi(pred3, ref3),
            clipiqa(pred3).mean(),
            compute_frr(pred, ref, bicubic_input).mean(),
            compute_frr(pred, ref, bicubic_input, fmin=0.0, fmax=0.8).mean(),
            compute_frr(pred, ref, bicubic_input, fmin=0.8, fmax=1.0).mean(),
            (
                torch.clip(
                    compute_frr(pred, ref, bicubic_input, fmin=0.8, fmax=1.0)
                    - compute_frr(pred, ref, bicubic_input, fmin=0.6, fmax=0.8),
                    min=0.0,
                )
            ).mean(),
        ]
    )


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="Use CARN network to for test step"
    )

    parser.add_argument(
        "--dataset", "-ds", type=str, help="Path to the dataset", required=True
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Path to output folder", required=True
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for sample patches selection"
    )

    parser.add_argument(
        "--crop", type=int, default=4, help="crop for metrics and prints"
    )
    parser.add_argument(
        "--nb_patches", default=128, type=int, help="Nb patches for benchmarking"
    )
    parser.add_argument("--band", type=int, default=2, help="band for metric study")

    return parser


def main():
    """
    Main method
    """
    # Parser arguments
    args = get_parser().parse_args()

    pl.seed_everything(args.seed)

    single_site_cfg = dataset.Sen2VnsSingleSiteDatasetConfig(
        load_20m_data=True, load_10m_data=True, load_b11b12=False
    )
    datamodule_cfg = dataset.Sen2VnsDataModuleConfig(
        sites=[
            "ARM",
            "BAMBENW2",
            "BENGA",
            "ESGISB-2",
            "ESGISB-1",
            "ESGISB-3",
            "ESTUAMAR",
            "FR-BIL",
            "FR-LAM",
            "ALSACE",
            "KUDALIAR",
            "LERIDA-1",
            "NARYN",
            "SO1",
            "SUDOUE-2",
            "SUDOUE-3",
            "SUDOUE-4",
            "SUDOUE-6",
            "JAM2018",
            "SUDOUE-5",
            "FR-LQ1",
            "ES-IC3XG",
            "ANJI",
            "MAD-AMBO",
            "ATTO",
            "ES-LTERA",
            "SO2",
        ],
        testing_sites=[
            "SO2",
        ],
        single_site_config=single_site_cfg,
        dataset_folder=args.dataset,
        batch_size=args.nb_patches,
        num_workers=4,
    )

    dataloader = dataset.Sen2VnsDataModule(datamodule_cfg).train_dataloader()

    batch = next(iter(dataloader))

    vns_tensor = batch.target.to(dtype=torch.float32, device="cuda")
    vns_tensor /= 10000.0
    s2_tensor = batch.network_input.hr_tensor.to(dtype=torch.float32, device="cuda")
    s2_tensor /= 10000.0

    s2_tensor = torch.nn.functional.interpolate(
        s2_tensor, scale_factor=2.0, mode="bicubic", align_corners=False
    )

    spatial_distorsion_values = np.arange(0.0, 2.2, 0.2)
    # mtf_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.6, None]
    mtf_values = [0.001, 0.01, 0.1, 0.2, 0.4, None]
    spectral_distorsion_values = np.arange(0.02, 0.12, 0.01)
    noise_values = np.arange(0.00, 0.01, 0.00025)
    pattern_values = np.arange(0.00, 0.01, 0.00025)

    fft_profiles: list[torch.Tensor] = []
    freqs: torch.Tensor
    for mtf in mtf_values:
        vns_sim = simulation.simulate(
            vns_tensor.clone(),
            mtf=mtf,
            spectral_distorsion=None,
            spatial_distorsion=None,
            noise_std=None,  # noise_values[1] if mtf is not None else 0,
            periodic_pattern=None,
            device="cuda",
        )

        _, freqs, prof = fda.compute_fft_profile(vns_sim)
        fft_profiles.append(prof)

    _, _, s2_profile = fda.compute_fft_profile(s2_tensor)

    plt.rcParams.update({"font.size": 14})
    ax: np.ndarray | Axes
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        freqs.cpu().numpy(),
        10 * np.log10(fft_profiles[-1][0, :, args.band].cpu().numpy())
        - 10 * np.log10(fft_profiles[-1][0, 1, args.band].cpu().numpy()),
        label="Venµs",
    )
    for i in range(len(mtf_values) - 1):
        ax.plot(
            freqs.cpu().numpy(),
            10 * np.log10(fft_profiles[i][0, :, args.band].cpu().numpy())
            - 10 * np.log10(fft_profiles[i][0, 1, args.band].cpu().numpy()),
            label=f"mtf={mtf_values[i]}",
            linestyle="--",
            alpha=0.25 + 0.75 * float(i) / len(mtf_values),
        )
    ax.plot(
        freqs.cpu().numpy(),
        10 * np.log10(s2_profile[0, :, args.band].cpu().numpy())
        - 10 * np.log10(s2_profile[0, 1, args.band].cpu().numpy()),
        label="Sentinel-2 (x2, bicubic)",
        color="green",
    )
    ax.fill_between(
        freqs.cpu().numpy(),
        10 * np.log10(s2_profile[0, :, args.band].cpu().numpy())
        - 10 * np.log10(s2_profile[0, 1, args.band].cpu().numpy()),
        10 * np.log10(fft_profiles[-1][0, :, args.band].cpu().numpy())
        - 10 * np.log10(fft_profiles[-1][0, 1, args.band].cpu().numpy()),
        color="green",
        alpha=0.2,
        label="frequencies restoration",
    )

    ax.legend()
    ax.grid(True)
    ax.set_ylabel("Attenuation (dB)")
    ax.set_xlabel("Spatial freq. f (1/px)")

    out_pdf = os.path.join(args.output, "fft_profile.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")

    metrics = []
    for spatial_distorsion in tqdm(
        spatial_distorsion_values,
        total=len(spatial_distorsion_values),
        desc="Computing spatial distorsion metrics",
    ):
        current_metrics = []
        for mtf in mtf_values:
            vns_sim = simulation.simulate(
                vns_tensor.clone(),
                mtf=mtf,
                spectral_distorsion=None,
                spatial_distorsion=spatial_distorsion,
                noise_std=None,
                periodic_pattern=None,
                device="cuda",
            )

            vns_sim = vns_sim[:, :, args.crop : -args.crop, args.crop : -args.crop]

            current_metrics.append(
                compute_metrics(
                    vns_sim[:, args.band : args.band + 1, ...],
                    vns_tensor[
                        :,
                        args.band : args.band + 1,
                        args.crop : -args.crop,
                        args.crop : -args.crop,
                    ],
                    s2_tensor[
                        :,
                        args.band : args.band + 1,
                        args.crop : -args.crop,
                        args.crop : -args.crop,
                    ],
                )
            )
        metrics.append(torch.stack(current_metrics, dim=0))
    spatial_distorsion_metrics = torch.stack(metrics, dim=0)

    metrics = []
    for spectral_distorsion in tqdm(
        spectral_distorsion_values,
        total=len(spectral_distorsion_values),
        desc="Computing spectral distorsion metrics",
    ):
        current_metrics = []
        for mtf in mtf_values:
            vns_sim = simulation.simulate(
                vns_tensor.clone(),
                mtf=mtf,
                spatial_distorsion=None,
                spectral_distorsion=spectral_distorsion,
                noise_std=None,
                periodic_pattern=None,
                device="cuda",
            )

            vns_sim = vns_sim[:, :, args.crop : -args.crop, args.crop : -args.crop]

            current_metrics.append(
                compute_metrics(
                    vns_sim[:, args.band : args.band + 1, ...],
                    vns_tensor[
                        :,
                        args.band : args.band + 1,
                        args.crop : -args.crop,
                        args.crop : -args.crop,
                    ],
                    s2_tensor[
                        :,
                        args.band : args.band + 1,
                        args.crop : -args.crop,
                        args.crop : -args.crop,
                    ],
                )
            )
        metrics.append(torch.stack(current_metrics, dim=0))
    spectral_distorsion_metrics = torch.stack(metrics, dim=0)

    metrics = []
    for noise_std in tqdm(
        noise_values,
        total=len(noise_values),
        desc="Computing noise metrics",
    ):
        current_metrics = []
        for mtf in mtf_values:
            vns_sim = simulation.simulate(
                vns_tensor.clone(),
                mtf=mtf,
                spatial_distorsion=None,
                spectral_distorsion=None,
                noise_std=noise_std,
                periodic_pattern=None,
                device="cuda",
            )

            vns_sim = vns_sim[:, :, args.crop : -args.crop, args.crop : -args.crop]

            current_metrics.append(
                compute_metrics(
                    vns_sim[:, args.band : args.band + 1, ...],
                    vns_tensor[
                        :,
                        args.band : args.band + 1,
                        args.crop : -args.crop,
                        args.crop : -args.crop,
                    ],
                    s2_tensor[
                        :,
                        args.band : args.band + 1,
                        args.crop : -args.crop,
                        args.crop : -args.crop,
                    ],
                )
            )
        metrics.append(torch.stack(current_metrics, dim=0))
    noise_metrics = torch.stack(metrics, dim=0)

    metrics = []
    for pattern in tqdm(
        pattern_values,
        total=len(pattern_values),
        desc="Computing pattern metrics",
    ):
        current_metrics = []
        for mtf in mtf_values:
            vns_sim = simulation.simulate(
                vns_tensor.clone(),
                mtf=mtf,
                spatial_distorsion=None,
                spectral_distorsion=None,
                noise_std=None,
                periodic_pattern=pattern,
                device="cuda",
            )

            vns_sim = vns_sim[:, :, args.crop : -args.crop, args.crop : -args.crop]

            current_metrics.append(
                compute_metrics(
                    vns_sim[:, args.band : args.band + 1, ...],
                    vns_tensor[
                        :,
                        args.band : args.band + 1,
                        args.crop : -args.crop,
                        args.crop : -args.crop,
                    ],
                    s2_tensor[
                        :,
                        args.band : args.band + 1,
                        args.crop : -args.crop,
                        args.crop : -args.crop,
                    ],
                )
            )
        metrics.append(torch.stack(current_metrics, dim=0))
    pattern_metrics = torch.stack(metrics, dim=0)

    for metric_idx, metric in enumerate(
        [
            "PSNR",
            "SSIM",
            "MS-SSIM",
            "IW-SSIM",
            "GMSD",
            "MS-GMSD",
            "BRISQUE",
            "BRISQUE_DIFF",
            "TV_DIFF",
            "DSS",
            "HAAR-PSI",
            "L1",
            "L2",
            "VIF_P",
            "FSIM",
            "SRSIM",
            "MDSI",
            "LPIPS",
            "Pie_APP",
            "DISTS",
            "VSI",
            "CLIP-IQA",
            "FRR",
            "FRR_0_80",
            "FRR_80_100",
            "FRR_Noise",
        ]
    ):
        fig, ax = plt.subplots(
            nrows=2, ncols=2, figsize=(10, 10), constrained_layout=True
        )
        assert isinstance(ax, np.ndarray)
        plt.rcParams.update({"font.size": 16})
        ax[0, 0].set_xlabel("Spatial distorsion (pixel)")
        ax[0, 0].set_ylabel(metric)
        ax[0, 0].grid(True)
        ax[1, 0].set_xlabel("Spectral distorsion (%)")
        ax[1, 0].set_ylabel(metric)
        ax[1, 0].grid(True)

        ax[0, 1].set_xlabel("Noise std (refl)")
        ax[0, 1].set_ylabel(metric)
        ax[0, 1].grid(True)
        ax[1, 1].set_xlabel("Pattern level")
        ax[1, 1].set_ylabel(metric)
        ax[1, 1].grid(True)

        for i, mtf in enumerate(mtf_values):
            ax[0, 0].plot(
                spatial_distorsion_values,
                spatial_distorsion_metrics[:, i, metric_idx].cpu().numpy(),
                label=f"mtf={mtf}",
                linestyle="--",
                linewidth=2,
                #                alpha=0.25 + 0.75 * float(i) / len(mtf_values),
            )
            ax[1, 0].plot(
                100 * spectral_distorsion_values,
                spectral_distorsion_metrics[:, i, metric_idx].cpu().numpy(),
                label=f"mtf={mtf}",
                linestyle="--",
                linewidth=2,
                #                alpha=0.25 + 0.75 * float(i) / len(mtf_values),
            )

            ax[0, 1].plot(
                noise_values,
                noise_metrics[:, i, metric_idx].cpu().numpy(),
                label=f"mtf={mtf}",
                linestyle="--",
                linewidth=2,
                #                alpha=0.25 + 0.75 * float(i) / len(mtf_values),
            )

            ax[1, 1].plot(
                pattern_values,
                pattern_metrics[:, i, metric_idx].cpu().numpy(),
                label=f"mtf={mtf}",
                linestyle="--",
                linewidth=2,
                #                alpha=0.25 + 0.75 * float(i) / len(mtf_values),
            )

        ax[0, 0].legend()
        ax[1, 0].legend()
        ax[0, 1].legend()
        ax[1, 1].legend()

        out_pdf = os.path.join(args.output, f"{metric}.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
