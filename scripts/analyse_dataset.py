#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from torchsisr import hydra_utils, patches
from torchsisr.dataset import batch_to_millirefl, generic_downscale
from torchsisr.fda import compute_fft_profile, compute_frr
from torchsisr.loss import PerBandBRISQUE


@hydra.main(version_base=None, config_path="../hydra/", config_name="main.yaml")
def main(config: DictConfig):
    # Call extras
    hydra_utils.extras(config)
    pl.seed_everything(config.seed, workers=True)
    # load samples
    train_dataloader = hydra.utils.instantiate(
        config.datamodule.data_module
    ).train_dataloader()

    # Estimate on that amount on batch
    nb_batches = 100
    model = hydra.utils.instantiate(config.training_module.training_module)
    iter_dataloader = iter(train_dataloader)

    if config.load_registration_checkpoint is not None:
        print(
            f"Restoring registration_module from {config.load_registration_checkpoint}"
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
            model.registration_module.load_state_dict(registration_module_parameters)

    brisque = PerBandBRISQUE()
    warp_losses: list[float] = []
    input_fft_profiles: list[torch.Tensor] = []
    target_fft_profiles: list[torch.Tensor] = []
    downscaled_diffs_hr: list[torch.Tensor] = []
    downscaled_diffs_lr: list[torch.Tensor] = []
    brisque_scores: list[torch.Tensor] = []
    for _ in tqdm(range(nb_batches), total=nb_batches, desc="Collecting samples"):
        batch = next(iter_dataloader)
        batch = batch_to_millirefl(batch, dtype=torch.float32)
        batch_std = model.standardize_batch(batch)

        batch_std = model.register_target(batch_std, radiometric_registration=True)
        batch.target = patches.unstandardize(
            batch_std.target,
            model.mean[:8],
            model.std[:8],
        )

        # batch_std = model.standardize_batch(batch)

        brisque_scores.append(brisque(batch.target, None))
        hr_scale_factor = (
            batch_std.target.shape[-1] / batch_std.network_input.hr_tensor.shape[-1]
        )
        downscaled_target_hr = generic_downscale(
            batch.target[:, :4, ...], factor=hr_scale_factor, mtf=0.1, padding="valid"
        )
        downscaled_diffs_hr.append(downscaled_target_hr - batch.network_input.hr_tensor)

        if batch.network_input.lr_tensor is not None:
            lr_scale_factor = (
                batch_std.target.shape[-1] / batch_std.network_input.lr_tensor.shape[-1]
            )
            downscaled_target_lr = generic_downscale(
                batch.target[:, 4:8, ...],
                factor=lr_scale_factor,
                mtf=0.1,
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

    downscaled_diffs_hr_rmse = torch.cat(downscaled_diffs_hr).std(dim=(0, 2, 3))
    downscaled_diffs_hr_bias = torch.cat(downscaled_diffs_hr).mean(dim=(0, 2, 3))
    brisque_scores = torch.stack(brisque_scores).mean(0)

    print("| Band \t | PRR \t | Radio \t | BRISQUE |")
    for band_id, band in enumerate(model.config.standardization.bands[:8]):
        _, _, prr = compute_frr(
            input_logprof[:, band_id],
            target_logprof[:, band_id],
            input_logprof[:, band_id],
        )
        print(
            f"| {band} \t | {prr:.2f}% \t |\
            {1000*downscaled_diffs_hr_bias[band_id]:.3f} \
            pm {1000*downscaled_diffs_hr_rmse[band_id]:.3f} \t \
            | {brisque_scores[band_id].item():.2f} |"
        )

    print(
        f"Geometric distortion amplitude mean: \
        {hr_scale_factor * torch.stack(warp_losses).mean().item():.3f} pix."
    )
    print(
        f"Geometric distortion amplitude std:\
        {hr_scale_factor * torch.stack(warp_losses).std().item():.3f} pix."
    )


if __name__ == "__main__":
    main()
