#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from torchsisr import dataset, hydra_utils
from torchsisr.dataset import batch_to_millirefl, generic_downscale
from torchsisr.fda import compute_fft_profile, compute_frr, plot_fft_profile


def build_default_dict(angle: float, nb_patches: int):
    return {"zenith_angle": angle, "nb_patches": nb_patches}


sites = {
    "FR-LQ1": build_default_dict(1.795402, 4888),
    "NARYN": build_default_dict(5.010906, 3814),
    "FGMANAUS": build_default_dict(7.232127, 129),
    "MAD-AMBO": build_default_dict(14.788115, 1443),
    "ARM": build_default_dict(15.160683, 15859),
    "BAMBENW2": build_default_dict(17.766533, 9018),
    "ES-IC3XG": build_default_dict(18.807686, 8823),
    "ANJI": build_default_dict(19.310494, 2314),
    "ATTO": build_default_dict(22.048651, 2258),
    "ESGISB-3": build_default_dict(23.683871, 6057),
    "ESGISB-1": build_default_dict(24.561609, 2892),
    "FR-BIL": build_default_dict(24.802892, 7105),
    "K34-AMAZ": build_default_dict(24.982675, 1385),
    "ESGISB-2": build_default_dict(26.209776, 3067),
    "ALSACE": build_default_dict(26.877071, 2654),
    "LERIDA-1": build_default_dict(28.524780, 2281),
    "ESTUAMAR": build_default_dict(28.871947, 912),
    "SUDOUE-5": build_default_dict(29.170244, 2176),
    "KUDALIAR": build_default_dict(29.180855, 7269),
    "SUDOUE-6": build_default_dict(29.192055, 2435),
    "SUDOUE-4": build_default_dict(29.516127, 935),
    "SUDOUE-3": build_default_dict(29.998115, 5363),
    "SO1": build_default_dict(30.255978, 12018),
    "SUDOUE-2": build_default_dict(31.295256, 9700),
    "ES-LTERA": build_default_dict(31.971764, 1701),
    "FR-LAM": build_default_dict(32.054056, 7299),
    "SO2": build_default_dict(32.218481, 738),
    "BENGA": build_default_dict(32.587334, 5858),
    "JAM2018": build_default_dict(33.718953, 2564),
}


@hydra.main(version_base=None, config_path="../hydra/", config_name="main.yaml")
def main(config: DictConfig):
    # Call extras
    hydra_utils.extras(config)
    pl.seed_everything(config.seed, workers=True)
    # load samples

    for site in sites.keys():
        datamodule_cfg = hydra.utils.instantiate(config.datamodule.config, sites=[site])

        datamodule = dataset.Sen2VnsDataModule(datamodule_cfg)

        # First, run on training set
        training_dataloader = datamodule.train_dataloader()

        # Estimate on that amount on batch
        nb_batches = min(20, len(training_dataloader))
        model = hydra.utils.instantiate(config.training_module.training_module)
        iter_dataloader = iter(training_dataloader)

        if config.load_registration_checkpoint is not None:
            print(
                f"Restoring registration_module from \
                {config.load_registration_checkpoint}"
            )
            checkpoint = torch.load(
                config.load_registration_checkpoint, map_location=torch.device("cpu")
            )
            registration_module_parameters = {
                k.split(".", maxsplit=1)[1]: v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("registration_module")
            }
            if registration_module_parameters:
                model.registration_module.load_state_dict(
                    registration_module_parameters
                )

        warp_losses: list[float] = []
        input_fft_profiles: list[torch.Tensor] = []
        target_fft_profiles: list[torch.Tensor] = []
        downscaled_diffs_hr: list[torch.Tensor] = []
        downscaled_diffs_lr: list[torch.Tensor] = []
        for _ in tqdm(
            range(nb_batches),
            total=nb_batches,
            desc=f"Collecting samples for site {site}",
        ):
            batch = next(iter_dataloader)
            batch = batch_to_millirefl(batch, dtype=torch.float32)
            batch_std = model.standardize_batch(batch)
        # batch_std = model.register_target(batch_std, radiometric_registration=True)
        # batch.target = patches.unstandardize(
        #     batch_std.target,
        #     model.mean[:8],
        #     model.std[:8],
        # )

        # batch_std = model.standardize_batch(batch)

        hr_scale_factor = (
            batch_std.target.shape[-1] / batch_std.network_input.hr_tensor.shape[-1]
        )
        downscaled_target_hr = generic_downscale(
            batch.target[:, :4, ...], factor=hr_scale_factor, mtf=0.4, padding="valid"
        )
        downscaled_diffs_hr.append(downscaled_target_hr - batch.network_input.hr_tensor)

        if batch.network_input.lr_tensor is not None:
            lr_scale_factor = (
                batch_std.target.shape[-1] / batch_std.network_input.lr_tensor.shape[-1]
            )
            downscaled_target_lr = generic_downscale(
                batch.target[:, 4:8, ...],
                factor=lr_scale_factor,
                mtf=0.4,
                padding="valid",
            )
            downscaled_diffs_lr.append(
                downscaled_target_lr - batch.network_input.lr_tensor
            )

        downscaled_diffs_hr.append(downscaled_target_hr - batch.network_input.hr_tensor)

        target_registration_band = generic_downscale(
            batch_std.target, factor=hr_scale_factor, mtf=0.4, padding="valid"
        )[:, model.config.registration.registration_channel, ...]
        ref_registration_band = batch_std.network_input.hr_tensor[
            :, model.config.registration.registration_channel, ...
        ]
        input_tensor = torch.nn.functional.interpolate(
            batch.network_input.hr_tensor,
            scale_factor=hr_scale_factor,
            mode="bicubic",
            align_corners=False,
        )
        if batch.network_input.lr_tensor is not None:
            lr_scale_factor = (
                batch_std.target.shape[-1] / batch_std.network_input.lr_tensor.shape[-1]
            )
            input_tensor = torch.cat(
                (
                    input_tensor,
                    torch.nn.functional.interpolate(
                        batch.network_input.lr_tensor,
                        scale_factor=lr_scale_factor,
                        mode="bicubic",
                        align_corners=False,
                    ),
                ),
                dim=1,
            )
        # Compute fft profiles
        _, freqs, profile = compute_fft_profile(
            input_tensor, s=2 * input_tensor.shape[-1]
        )
        input_fft_profiles.append(profile)
        _, _, profile = compute_fft_profile(batch.target, s=2 * input_tensor.shape[-1])
        target_fft_profiles.append(profile)

        with torch.no_grad():
            target_flow = model.registration_module(
                target_registration_band, ref_registration_band
            )
            target_warp_loss = torch.norm(target_flow, p=2, dim=1)
            warp_losses.append(target_warp_loss)

        input_fft_profile = torch.cat(input_fft_profiles, dim=0).mean(dim=0)
        target_fft_profile = torch.cat(target_fft_profiles, dim=0).mean(dim=0)

        target_logprof = 10 * torch.log10(target_fft_profile) - 10 * torch.log10(
            target_fft_profile[1:2, ...]
        )

        input_logprof = 10 * torch.log10(input_fft_profile) - 10 * torch.log10(
            input_fft_profile[1:2, ...]
        )

        for band_id, band in enumerate(model.config.standardization.bands[:8]):
            plot_fft_profile(
                torch.stack(
                    [input_fft_profile, target_fft_profile, input_fft_profile], dim=0
                ),
                freqs=freqs,
                output_pdf=f"fft_prof_{site}.pdf",
            )
            _, _, prr = compute_frr(
                input_logprof[:, band_id],
                target_logprof[:, band_id],
                input_logprof[:, band_id],
            )
            sites[site][f"prr_{band}"] = prr.item()
        downscaled_diffs_hr_rmse = torch.cat(downscaled_diffs_hr).std(dim=(0, 2, 3))
        downscaled_diffs_hr_bias = torch.cat(downscaled_diffs_hr).mean(dim=(0, 2, 3))
        for band_id, band in enumerate(model.config.standardization.bands[:4]):
            sites[site][f"rad_diff_bias_{band}"] = downscaled_diffs_hr_bias[
                band_id
            ].item()
            sites[site][f"rad_diff_rmse_{band}"] = downscaled_diffs_hr_rmse[
                band_id
            ].item()

        if downscaled_diffs_lr:
            downscaled_diffs_lr_rmse = torch.cat(downscaled_diffs_lr).std(dim=(0, 2, 3))
            downscaled_diffs_lr_bias = torch.cat(downscaled_diffs_lr).mean(
                dim=(0, 2, 3)
            )
            for band_id, band in enumerate(model.config.standardization.bands[4:8]):
                sites[site][f"rad_diff_bias_{band}"] = downscaled_diffs_lr_bias[
                    band_id
                ].item()
                sites[site][f"rad_diff_rmse_{band}"] = downscaled_diffs_lr_rmse[
                    band_id
                ].item()

        sites[site]["geom_diff_amp_mean"] = torch.stack(warp_losses).mean().item()
        sites[site]["geom_diff_amp_std"] = torch.stack(warp_losses).std().item()

    sites_df = pd.DataFrame.from_dict(sites, orient="index")
    sites_df.to_csv("s2v_site_analysis.csv", sep="\t")


if __name__ == "__main__":
    main()
