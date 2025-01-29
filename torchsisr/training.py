# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the pytorch lightining module allowing to
train the super-resolution networks
"""
import logging
import os
import random
import typing
from dataclasses import dataclass
from enum import Enum
from itertools import chain

import numpy as np
import pytorch_lightning as pl
import torch
from sensorsio.sentinel2 import Sentinel2

from torchsisr import patches
from torchsisr.custom_types import BatchData, NetworkInput, PredictedData
from torchsisr.dataset import (
    align_min_max_batch,
    batch_to_millirefl,
    generate_psf_kernel,
    generic_downscale,
    match_bands,
    simulate_batch,
    wald_batch,
)
from torchsisr.discriminator import Discriminator
from torchsisr.double_sisr_model import DoubleSuperResolutionModel
from torchsisr.fda import (
    compute_fda_noise,
    compute_fft_profile,
    compute_fro_fru,
    compute_frr,
    plot_fft_profile,
)
from torchsisr.loss_helper import PixelLossWrapper
from torchsisr.registration import (
    UnetOpticalFlowEstimation,
    compose_flows,
    simulate_warp,
    warp,
)


@dataclass
class StandardizationParameters:
    """
    Represents standardization parameters
    """

    bands: tuple[Sentinel2.Band, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self):
        if isinstance(self.bands[0], str):
            object.__setattr__(
                self, "bands", tuple(Sentinel2.Band(b) for b in self.bands)
            )


@dataclass
class BatchSimulationParameters:
    """
    Parameters for batch simulation
    """

    mtf_min: float = 0.1
    mtf_max: float | None = None
    noise_std: tuple[float, ...] | None = None
    noise_multiplier_min: float = 1.0
    noise_multiplier_max: float | None = None


@dataclass
class OptimizationParameters:
    """
    Dataclass for optimization parameters
    """

    learning_rate: float
    t_0: int
    t_mult: float


class AdversarialLoss(Enum):
    """
    Enum for adversarial formulation types
    """

    BCE = "BCE"
    DRA_BCE = "DRA_BCE"
    WASSERSTEIN = "WASSERSTEIN"


@dataclass
class RegistrationParameters:
    """
    Dataclass for registration parameters
    """

    registration_channel: int = 3
    max_offset: float = 2.0
    depth: int = 4
    min_skip_depth: int = 2
    nb_features: int = 64


@dataclass
class AdversarialParameters:
    """
    Dataclass for adversarial parameters
    """

    bands: tuple[Sentinel2.Band, ...]
    discriminator: Discriminator
    optimization: OptimizationParameters
    starting_step_discriminator: int = 0
    starting_step_generator: int = 0
    weight: float = 0.1
    mode: str | AdversarialLoss = "WASSERSTEIN"
    real_label_smoothing: float = 0.1

    def __post_init__(self):
        if isinstance(self.bands[0], str):
            self.bands = tuple(Sentinel2.Band(b) for b in self.bands)

        if isinstance(self.mode, str):
            self.mode = AdversarialLoss(self.mode)


@dataclass
class WaldParameters:
    """
    Dataclass for Wald simulation parameters
    """

    losses: tuple[PixelLossWrapper, ...]
    validation_metrics: tuple[PixelLossWrapper, ...]
    test_metrics: tuple[PixelLossWrapper, ...]
    validation_margin: int
    noise_std: tuple[float, ...] | None = None
    noise_multiplier_min: float = 0.1
    noise_multiplier_max: float | None = None
    mtf_min: float = 0.1
    mtf_max: float | None = None
    pad_to_input_size: bool = True


@dataclass
class DoubleSISRTrainingModuleConfig:
    """
    Full configuration of training module
    """

    model: DoubleSuperResolutionModel  # The model to train
    optimization: OptimizationParameters
    standardization: StandardizationParameters  # The standardization parameters
    batch_simulation: BatchSimulationParameters  # The batch simulation parameters
    real_losses: tuple[PixelLossWrapper, ...]  # Data fidelity losses
    sim_losses: tuple[PixelLossWrapper, ...] | None
    validation_metrics: tuple[
        PixelLossWrapper, ...
    ]  # metrics (only used in validation)
    validation_margin: int  # Crop margin for validation
    align_min_max: bool
    test_metrics: tuple[PixelLossWrapper, ...]  # Metrics for testing
    adversarial: None | AdversarialParameters = (
        None  # Parameters for adversarial training
    )
    wald: None | WaldParameters = None  # Parameters for Wald training
    registration: None | RegistrationParameters = None
    training_geometric_registration: bool = False
    training_radiometric_registration: bool = False
    testval_geometric_registration: bool = False
    testval_radiometric_registration: bool = False
    pretrain_registration: bool = False
    warp_loss_weight: float | None = None


class DoubleSISRTrainingModule(pl.LightningModule):
    # pylint: disable=too-many-ancestors
    """
    Lightning wrapper for the training of DoubleSISRTrainingModule
    """

    def __init__(self, config: DoubleSISRTrainingModuleConfig):
        """
        Constructor
        """
        super().__init__()

        # In order to handle optional GAN loss
        self.automatic_optimization = False

        self.config = config

        # Store models as class members so that their parameters
        # are visible to pytorch lightning
        self.generator = self.config.model
        self.discriminator: Discriminator | None = None
        if self.config.adversarial is not None:
            self.discriminator = self.config.adversarial.discriminator

        self.register_buffer("mean", torch.tensor(self.config.standardization.mean))
        self.register_buffer("std", torch.tensor(self.config.standardization.std))
        if self.config.batch_simulation.noise_std is not None:
            self.register_buffer(
                "noise_std", torch.tensor(self.config.batch_simulation.noise_std)
            )
        else:
            self.noise_std: torch.Tensor | None = None

        # We need to store
        self.validation_metrics = torch.nn.ModuleList(self.config.validation_metrics)
        self.test_metrics = torch.nn.ModuleList(self.config.test_metrics)

        self.wald_validation_metrics: torch.nn.ModuleList | None = None
        self.wald_test_metrics: torch.nn.ModuleList | None = None
        self.wald_losses: torch.nn.ModuleList | None = None

        if self.config.wald is not None:
            if self.config.wald.noise_std is not None:
                self.register_buffer(
                    "wald_noise_std", torch.tensor(self.config.wald.noise_std)
                )
            else:
                self.wald_noise_std: torch.Tensor | None = None

            self.wald_validation_metrics = torch.nn.ModuleList(
                self.config.wald.validation_metrics
            )
            self.wald_test_metrics = torch.nn.ModuleList(self.config.wald.test_metrics)
            self.wald_losses = torch.nn.ModuleList(self.config.wald.losses)
        self.real_losses = torch.nn.ModuleList(self.config.real_losses)
        self.sim_losses: torch.nn.ModuleList | None = None
        if self.config.sim_losses:
            self.sim_losses = torch.nn.ModuleList(self.config.sim_losses)

        if self.config.registration is not None:
            self.registration_module = UnetOpticalFlowEstimation(
                depth=self.config.registration.depth,
                min_skip_depth=self.config.registration.min_skip_depth,
                start_filts=self.config.registration.nb_features,
                max_range=self.config.registration.max_offset,
            )

        self.input_fft_profiles: list[torch.Tensor] = []
        self.target_fft_profiles: list[torch.Tensor] = []
        self.pred_fft_profiles: list[torch.Tensor] = []
        self.fft_profiles_frequencies: torch.Tensor

    def get_main_optimizer_and_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """
        Return main optimizer and scheduler
        """
        optimizers = self.optimizers()
        if isinstance(optimizers, list):

            return optimizers[0], self.lr_schedulers()[0]  # type: ignore
        return self.optimizers(), self.lr_schedulers()  # type: ignore

    def get_adversarial_optimizer_and_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """
        Return adversarial optimizer and sheduler
        """
        optimizers: list[pl.core.optimizer.LightningOptimizer] = typing.cast(
            list[pl.core.optimizer.LightningOptimizer], self.optimizers()
        )

        assert len(optimizers) > 1
        assert len(self.lr_schedulers()) > 1  # type: ignore
        return optimizers[1], self.lr_schedulers()[1]  # type: ignore

    def standardize_batch(self, batch: BatchData) -> BatchData:
        """
        Standardize batch
        """
        # Find out if we standardize or unstandardize
        standardization_fn = patches.standardize

        def _internal_call(
            input_tensor: torch.Tensor, input_bands: tuple[Sentinel2.Band, ...]
        ) -> torch.Tensor:
            """
            Factors a few lines of code
            """
            bands, _ = match_bands(input_bands, self.config.standardization.bands)
            return standardization_fn(
                input_tensor,
                self.mean[list(bands)],
                self.std[list(bands)],
            )

        std_hr_input = _internal_call(
            batch.network_input.hr_tensor, batch.network_input.hr_bands
        )
        std_target = _internal_call(batch.target, batch.target_bands)

        # Handle optional lr input
        std_lr_input: torch.Tensor | None = None
        if (
            batch.network_input.lr_tensor is not None
            and batch.network_input.lr_bands is not None
        ):
            std_lr_input = _internal_call(
                batch.network_input.lr_tensor, batch.network_input.lr_bands
            )

        # Build output batch data
        return BatchData(
            NetworkInput(
                std_hr_input,
                batch.network_input.hr_bands,
                std_lr_input,
                batch.network_input.lr_bands,
            ),
            std_target,
            batch.target_bands,
        )

    def unstandardize_pred(self, prediction: PredictedData) -> PredictedData:
        """
        Unstandardize predictions
        """
        bands, _ = match_bands(prediction.bands, self.config.standardization.bands)
        pred_ustd = patches.unstandardize(
            prediction.prediction,
            self.mean[list(bands)],
            self.std[list(bands)],
        )

        return PredictedData(pred_ustd, prediction.bands, prediction.margin)

    def simulate_batch(self, batch: BatchData) -> BatchData:
        """
        Apply batch simulation with randomness
        """
        mtf = self.config.batch_simulation.mtf_min
        if self.config.batch_simulation.mtf_max is not None:
            mtf = random.uniform(
                self.config.batch_simulation.mtf_min,
                self.config.batch_simulation.mtf_max,
            )
        std_multiplier = self.config.batch_simulation.noise_multiplier_min
        if self.config.batch_simulation.noise_multiplier_max is not None:
            std_multiplier = random.uniform(
                self.config.batch_simulation.noise_multiplier_min,
                self.config.batch_simulation.noise_multiplier_max,
            )

        batch = simulate_batch(
            batch,
            mtf=mtf,
            noise_std=(
                None if self.noise_std is None else std_multiplier * self.noise_std
            ),
        )
        return batch

    def eval_pixel_losses_and_metrics(
        self,
        batch_data: BatchData,
        network_output: PredictedData,
        losses: torch.nn.ModuleList,
        margin: int | None = None,
        context: str = "training",
        average: bool = False,
    ) -> torch.Tensor:
        """
        Sequential evaluation of all losses
        """
        assert batch_data.target.device == self.device
        assert batch_data.network_input.hr_tensor.device == self.device
        if batch_data.network_input.lr_tensor is not None:
            assert batch_data.network_input.lr_tensor.device == self.device
        assert network_output.prediction.device == self.device

        total_loss = torch.zeros((1,), device=self.device)

        for loss in losses:
            loss_output = loss(batch_data, network_output, margin)

            if context.startswith("test"):
                for band_id, band in enumerate(loss_output.bands):
                    loss_key = context + "_per_band/" + loss.name + "_" + band.value
                    self.log(
                        loss_key,
                        loss_output.loss_values[band_id],
                        batch_size=batch_data.network_input.hr_tensor.shape[0],
                    )

            if not average:
                current_total_loss = loss_output.loss_values.sum()
                label = "sum"
            else:
                current_total_loss = loss_output.loss_values.mean()
                label = "average"
            self.log(
                context + "/" + loss.name + "_" + label,
                current_total_loss,
                batch_size=batch_data.network_input.hr_tensor.shape[0],
            )
            total_loss += current_total_loss

        return total_loss

    def relativistic_average_discriminator(
        self, tensor_real: torch.Tensor, tensor_fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Implement relativistic average discriminator as in ESRGAN Paper
        """
        assert self.discriminator
        disc_real = self.discriminator(tensor_real)
        disc_fake_avg = self.discriminator(tensor_fake).mean(dim=0, keepdim=True)

        return disc_real - disc_fake_avg

    def adversarial_generator_loss(
        self, tensor_real: torch.Tensor, tensor_fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Implement adversarial loss from ESRGAN paper
        """
        assert self.discriminator
        assert self.config.adversarial
        match self.config.adversarial.mode:
            case AdversarialLoss.WASSERSTEIN:
                disc_fake = self.discriminator(tensor_fake)
                return disc_fake.mean()
            case AdversarialLoss.BCE:
                disc_fake = self.discriminator(tensor_fake)
                target_labels_fake = torch.ones_like(disc_fake)

                adv_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(
                    disc_fake, target_labels_fake
                )
                return adv_loss_fake

            case AdversarialLoss.DRA_BCE:
                dra_real = self.relativistic_average_discriminator(
                    tensor_real, tensor_fake
                )
                # The out of order here is intended
                # pylint: disable=arguments-out-of-order
                dra_fake = self.relativistic_average_discriminator(
                    tensor_fake, tensor_real
                )

                target_labels_real = torch.zeros_like(dra_real)
                target_labels_fake = torch.ones_like(dra_fake)

                adv_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(
                    dra_real, target_labels_real
                )
                adv_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(
                    dra_fake, target_labels_fake
                )

                adv_loss = 0.5 * (adv_loss_real + adv_loss_fake)
                return adv_loss
        raise NotImplementedError

    def adversarial_discriminator_loss(
        self, tensor_real: torch.Tensor, tensor_fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Implement adversarial loss from ESRGAN paper
        """
        assert self.config.adversarial
        assert self.discriminator
        match self.config.adversarial.mode:
            case AdversarialLoss.WASSERSTEIN:
                disc_fake = self.discriminator(tensor_fake.detach())
                disc_real = self.discriminator(tensor_real)

                adv_loss = disc_real.mean() - disc_fake.mean()
                return adv_loss

            case AdversarialLoss.DRA_BCE:
                dra_real = self.relativistic_average_discriminator(
                    tensor_real, tensor_fake.detach()
                )
                dra_fake = self.relativistic_average_discriminator(
                    tensor_fake.detach(), tensor_real
                )

                target_labels_real = (
                    torch.ones_like(dra_real)
                    - self.config.adversarial.real_label_smoothing
                )
                target_labels_fake = torch.zeros_like(dra_fake)

                adv_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(
                    dra_real, target_labels_real
                )
                adv_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(
                    dra_fake, target_labels_fake
                )

                adv_loss = 0.5 * (adv_loss_real + adv_loss_fake)
                return adv_loss
            case AdversarialLoss.BCE:
                disc_fake = self.discriminator(tensor_fake.detach())
                disc_real = self.discriminator(tensor_real)

                target_labels_real = (
                    torch.ones_like(disc_real)
                    - self.config.adversarial.real_label_smoothing
                )
                target_labels_fake = torch.zeros_like(disc_fake)

                adv_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(
                    disc_real, target_labels_real
                )
                adv_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(
                    disc_fake, target_labels_fake
                )

                adv_loss = 0.5 * (adv_loss_real + adv_loss_fake)
                return adv_loss

        raise NotImplementedError

    def adversarial_generator_training_step(
        self, batch_std: BatchData, pred_std: PredictedData
    ):
        """
        Training step of generator for aversarial training
        """
        assert self.config.adversarial
        # identify bands
        pred_bands, _ = match_bands(self.config.adversarial.bands, pred_std.bands)
        target_bands, _ = match_bands(
            self.config.adversarial.bands, batch_std.target_bands
        )

        adv_loss_gen = self.adversarial_generator_loss(
            batch_std.target[
                :,
                target_bands,
                pred_std.margin : -pred_std.margin,
                pred_std.margin : -pred_std.margin,
            ],
            pred_std.prediction[
                :,
                pred_bands,
                pred_std.margin : -pred_std.margin,
                pred_std.margin : -pred_std.margin,
            ],
        )
        # In that case we still need the value
        if self.global_step <= self.config.adversarial.starting_step_generator:
            adv_loss_gen = adv_loss_gen.detach()

        self.log(
            "training/generator_dra",
            adv_loss_gen,
            batch_size=pred_std.prediction.shape[0],
        )
        total_loss = self.config.adversarial.weight * adv_loss_gen

        return total_loss

    def adversarial_discriminator_training_step(
        self, batch_std: BatchData, pred_std: PredictedData
    ):
        """
        Training step for discriminator in adversarial training
        """
        assert self.config.adversarial
        if self.global_step > self.config.adversarial.starting_step_discriminator:
            # Get optimizer and scheduler
            d_optimizer, d_scheduler = self.get_adversarial_optimizer_and_scheduler()

            # Step scheduler for discriminator
            d_scheduler.step()

            # identify bands
            pred_bands, _ = match_bands(self.config.adversarial.bands, pred_std.bands)
            target_bands, _ = match_bands(
                self.config.adversarial.bands, batch_std.target_bands
            )
            # Generator adversarial loss
            self.toggle_optimizer(d_optimizer)

            adv_loss_disc = self.adversarial_discriminator_loss(
                batch_std.target[
                    :,
                    target_bands,
                    pred_std.margin : -pred_std.margin,
                    pred_std.margin : -pred_std.margin,
                ],
                pred_std.prediction[
                    :,
                    pred_bands,
                    pred_std.margin : -pred_std.margin,
                    pred_std.margin : -pred_std.margin,
                ],
            )

            self.log(
                "training/discriminator_dra",
                adv_loss_disc,
                batch_size=pred_std.prediction.shape[0],
            )

            self.manual_backward(adv_loss_disc)
            d_optimizer.step()
            d_optimizer.zero_grad()
            self.untoggle_optimizer(d_optimizer)

    def wald_batch(self, batch: BatchData) -> BatchData:
        """
        Apply wald batch simulation
        """
        assert self.config.wald
        mtf = self.config.wald.mtf_min

        if self.config.wald.mtf_max is not None:
            mtf = random.uniform(self.config.wald.mtf_min, self.config.wald.mtf_max)

        std_multiplier = self.config.wald.noise_multiplier_min

        if self.config.wald.noise_multiplier_max is not None:
            std_multiplier = random.uniform(
                self.config.wald.noise_multiplier_min,
                self.config.wald.noise_multiplier_max,
            )
        return wald_batch(
            batch,
            noise_std=(
                None
                if self.wald_noise_std is None
                else std_multiplier * self.wald_noise_std
            ),
            pad_to_input_size=self.config.wald.pad_to_input_size,
            mtf=mtf,
        )

    def wald_training_step(self, batch: BatchData) -> torch.Tensor:
        """
        Handle Wald step
        """

        # Apply wald simulation
        wald_sim_batch = self.wald_batch(batch)
        wald_sim_batch_std = self.standardize_batch(wald_sim_batch)

        # Call full res forward pass
        network_output = self.generator(wald_sim_batch_std.network_input)

        # Evaluate pixel losses
        assert self.wald_losses
        total_loss = self.eval_pixel_losses_and_metrics(
            wald_sim_batch_std,
            network_output,
            self.wald_losses,
            context="training",
            average=True,
        )

        return total_loss

    def predict(self, batch: BatchData, simulate: bool = False) -> PredictedData:
        """
        Simple prediction routine
        """
        # First convert batch to millirefl
        batch = batch_to_millirefl(batch, dtype=self.mean.dtype)

        # Align min / max per band on each patch
        if self.config.align_min_max:
            batch = align_min_max_batch(batch)

        if simulate:
            batch = self.simulate_batch(batch)

        # Apply standardization to batch
        batch_std = self.standardize_batch(batch)

        # Call full res forward pass
        with torch.no_grad():
            network_output = self.generator(batch_std.network_input)

        return self.unstandardize_pred(network_output)

    def registration_step(
        self, batch: BatchData, context: str = "training"
    ) -> torch.Tensor:
        """
        Registration module training
        """
        assert self.config.registration
        sim_batch = self.simulate_batch(batch)

        # First, learn to recover a simulated flow
        source_band = sim_batch.network_input.hr_tensor[
            :, self.config.registration.registration_channel, :, :
        ]
        real_target_band = batch.network_input.hr_tensor[
            :, self.config.registration.registration_channel, :, :
        ]
        target_flow = simulate_warp(
            source_band,
            max_range=2,
            max_width=10,
        )

        source_band_warped = warp(source_band[:, None, :, :], target_flow)[:, 0, :, :]

        flow1 = self.registration_module(source_band, real_target_band)
        flow1_bwd = self.registration_module(real_target_band, source_band)
        flow2 = self.registration_module(source_band_warped, real_target_band)

        flow_loss = torch.nn.functional.smooth_l1_loss(
            compose_flows(target_flow, flow2), flow1
        )
        self.log(f"{context}/registration_flow_loss", flow_loss)

        flow_bias = self.registration_module(source_band, source_band)

        flow_bias_loss = torch.nn.functional.smooth_l1_loss(
            flow_bias, torch.zeros_like(flow_bias)
        )
        self.log(f"{context}/registration_flow_bias_loss", flow_bias_loss)

        flow_consistency_loss = torch.nn.functional.smooth_l1_loss(
            compose_flows(flow1, flow1_bwd), torch.zeros_like(flow1)
        )
        self.log(f"{context}/registration_flow_consistency_loss", flow_consistency_loss)

        warped = warp(source_band[:, None, :, :], flow1)[:, 0, :, :]
        data_loss = torch.nn.functional.smooth_l1_loss(warped, real_target_band)

        self.log(f"{context}/registration_data_loss", data_loss)

        total_loss = flow_loss + flow_bias_loss + data_loss + flow_consistency_loss
        self.log(f"{context}/registration_total_loss", total_loss)

        return total_loss

    def register_target(
        self, batch_data: BatchData, radiometric_registration: bool = True
    ) -> BatchData:
        """
        Apply the registration algorithm to register target to source
        """
        assert self.registration_module
        assert self.config.registration

        for params in self.registration_module.parameters():
            params.requires_grad = False

        sim_batch = self.simulate_batch(batch_data)

        # First, learn to recover a simulated flow
        source_band = sim_batch.network_input.hr_tensor[
            :, self.config.registration.registration_channel, :, :
        ]
        target_band = batch_data.network_input.hr_tensor[
            :, self.config.registration.registration_channel, :, :
        ]

        flow_hr = self.registration_module(source_band, target_band)

        # Smooth out optical flow to avoid decreasing image quality
        psf_kernel = torch.tensor(
            generate_psf_kernel(1.0, 1.0, 0.000001, 7),
            device=flow_hr.device,
            dtype=flow_hr.dtype,
        )
        # pylint: disable=not-callable
        flow_hr = torch.nn.functional.conv2d(
            flow_hr,
            psf_kernel[None, None, :, :].expand(flow_hr.shape[1], -1, -1, -1),
            groups=flow_hr.shape[1],
            padding="same",
        )

        # Upsample estimated flow
        scale_factor = (
            sim_batch.target.shape[-1] / sim_batch.network_input.hr_tensor.shape[-1]
        )
        flow_x2 = scale_factor * torch.nn.functional.interpolate(
            flow_hr, scale_factor=scale_factor, align_corners=False, mode="bicubic"
        )

        target = warp(batch_data.target, flow_x2)

        if radiometric_registration:
            psf_kernel = torch.tensor(
                generate_psf_kernel(1.0, 1.0, 0.00001, 7),
                device=flow_hr.device,
                dtype=flow_hr.dtype,
            )

            warped_hr = warp(sim_batch.network_input.hr_tensor, flow_hr)
            warped_lr: torch.Tensor | None = None
            radio_corr_hr = torch.nn.functional.interpolate(
                # pylint: disable=not-callable
                torch.nn.functional.conv2d(
                    batch_data.network_input.hr_tensor - warped_hr,
                    psf_kernel[None, None, :, :].expand(warped_hr.shape[1], -1, -1, -1),
                    groups=warped_hr.shape[1],
                    padding="same",
                ),
                mode="bicubic",
                scale_factor=scale_factor,
                align_corners=False,
            )
            radiometric_correction = radio_corr_hr
            if sim_batch.network_input.lr_tensor is not None:
                assert batch_data.network_input.lr_tensor
                flow_lr = 0.5 * torch.nn.functional.interpolate(
                    flow_hr, mode="bicubic", scale_factor=0.5, align_corners=False
                )
                warped_lr = warp(sim_batch.network_input.lr_tensor, flow_lr)

                radio_corr_lr = torch.nn.functional.interpolate(
                    torch.nn.functional.conv2d(
                        batch_data.network_input.lr_tensor[:, :4, ...]
                        - warped_lr[:, :4, ...],
                        psf_kernel[None, None, :, :].expand(
                            warped_hr.shape[1], -1, -1, -1
                        ),
                        groups=warped_hr.shape[1],
                        padding="same",
                    ),
                    mode="bicubic",
                    scale_factor=2 * scale_factor,
                    align_corners=False,
                )
                radiometric_correction = torch.cat(
                    (radio_corr_hr, radio_corr_lr), dim=1
                )

            target = target + radiometric_correction

        batch_data.target = target
        return batch_data

    # pylint: disable=arguments-differ
    def training_step(self, batch: BatchData, _batch_idx: int):
        """
        Train iteration
        """
        # First convert batch to millirefl
        batch = batch_to_millirefl(batch, dtype=self.mean.dtype)

        # Align min / max per band on each patch
        if self.config.align_min_max:
            batch = align_min_max_batch(batch)

        optimizer, scheduler = self.get_main_optimizer_and_scheduler()

        # Activate model optimizer
        self.toggle_optimizer(optimizer)

        # Apply standardization to batch
        batch_std = self.standardize_batch(batch)

        sim_batch = self.simulate_batch(batch)
        sim_batch_std = self.standardize_batch(sim_batch)

        # Pretrain registration module
        if self.config.pretrain_registration:
            loss = self.registration_step(batch_std, context="training")
            scheduler.step()
            self.manual_backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            return loss

        if self.config.registration and self.config.training_geometric_registration:
            batch_std = self.register_target(
                batch_std, self.config.training_radiometric_registration
            )

        # Call full res forward pass
        network_output = self.generator(batch_std.network_input)

        # Evaluate pixel losses
        total_loss: torch.Tensor | None = None

        if self.config.registration and self.config.warp_loss_weight:
            for params in self.registration_module.parameters():
                params.requires_grad = False

            pred_registration_band = generic_downscale(
                network_output.prediction, factor=2.0, mtf=0.3, padding="valid"
            )[:, self.config.registration.registration_channel, ...]

            input_registration_band = batch_std.network_input.hr_tensor[
                :, self.config.registration.registration_channel, ...
            ]
            target_registration_band = sim_batch_std.network_input.hr_tensor[
                :, self.config.registration.registration_channel, ...
            ]
            pred_flow = self.registration_module(
                pred_registration_band, target_registration_band
            )
            ref_flow = self.registration_module(
                input_registration_band, target_registration_band
            )

            warp_loss = torch.nn.functional.smooth_l1_loss(pred_flow, ref_flow)

            self.log("training/average_input_warp", warp_loss)

            total_loss = self.config.warp_loss_weight * warp_loss[None]

        if self.real_losses is not None:
            if total_loss is None:
                total_loss = self.eval_pixel_losses_and_metrics(
                    batch_std,
                    network_output,
                    self.real_losses,
                    context="training",
                    average=True,
                )
            else:
                total_loss += self.eval_pixel_losses_and_metrics(
                    batch_std,
                    network_output,
                    self.real_losses,
                    context="training",
                    average=True,
                )
        if self.sim_losses is not None:
            network_output_sim = self.generator(sim_batch_std.network_input)
            # Evaluate pixel losses
            sim_total_loss = self.eval_pixel_losses_and_metrics(
                sim_batch_std,
                network_output_sim,
                self.sim_losses,
                context="training",
                average=True,
            )
            if total_loss is not None:
                total_loss += sim_total_loss
            else:
                total_loss = sim_total_loss

        # Perform Wald training if required
        if self.wald_losses is not None:
            wald_total_loss = self.wald_training_step(batch)

            if total_loss is not None:
                total_loss += wald_total_loss
            else:
                total_loss = wald_total_loss

        if total_loss is not None:
            self.log(
                "training/total_pixel_loss",
                total_loss,
                batch_size=batch.network_input.hr_tensor.shape[0],
            )

        # If we need to do adversarial training
        if self.config.adversarial is not None and self.discriminator is not None:
            gen_total_loss = self.adversarial_generator_training_step(
                batch_std, network_output
            )
            if total_loss is not None:
                total_loss += gen_total_loss
            else:
                total_loss = gen_total_loss

        if total_loss is not None:
            self.log(
                "training/total_loss",
                total_loss,
                batch_size=batch.network_input.hr_tensor.shape[0],
            )
            # Advance scheduler
            scheduler.step()
            self.manual_backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()
        self.untoggle_optimizer(optimizer)

        # If we need to do adversarial training
        if self.config.adversarial is not None and self.discriminator is not None:
            self.adversarial_discriminator_training_step(batch_std, network_output)

        return total_loss

    def wald_validation_test_step(
        self,
        batch: BatchData,
        metrics: torch.nn.ModuleList,
        context: str,
    ) -> torch.Tensor:
        """
        Perform Wald validation step if required
        """
        assert self.wald_losses
        assert self.config.wald
        # Apply wald simulation
        wald_sim_batch = self.wald_batch(batch)
        wald_sim_batch_std = self.standardize_batch(wald_sim_batch)

        # Call full res forward pass
        with torch.no_grad():
            network_output = self.generator(wald_sim_batch_std.network_input)
        network_output_ustd = self.unstandardize_pred(network_output)

        # Evaluate pixel losses

        total_loss = self.eval_pixel_losses_and_metrics(
            wald_sim_batch_std,
            network_output,
            self.wald_losses,
            context=context + "_losses",
            margin=self.config.wald.validation_margin,
            average=True,
        )

        # Evaluate pixel metrics
        self.eval_pixel_losses_and_metrics(
            wald_sim_batch,
            network_output_ustd,
            metrics,
            context=context + "_metrics",
            margin=self.config.wald.validation_margin,
            average=True,
        )

        return total_loss

    def adversarial_validation_test_step(
        self,
        batch_std: BatchData,
        out_std: PredictedData,
        context: str,
    ) -> torch.Tensor:
        """
        Generic adversarial test and validation step
        """
        assert self.config.adversarial
        # identify bands
        margin = self.config.validation_margin
        pred_bands, _ = match_bands(self.config.adversarial.bands, out_std.bands)
        target_bands, _ = match_bands(
            self.config.adversarial.bands, batch_std.target_bands
        )
        adv_loss_gen = self.adversarial_generator_loss(
            batch_std.target[
                :,
                target_bands,
                margin:-margin,
                margin:-margin,
            ],
            out_std.prediction[
                :,
                pred_bands,
                margin:-margin,
                margin:-margin,
            ],
        )

        self.log(
            context + "/generator_dra",
            adv_loss_gen,
            batch_size=out_std.prediction.shape[0],
        )

        adv_loss_disc = self.adversarial_discriminator_loss(
            batch_std.target[
                :,
                target_bands,
                margin:-margin,
                margin:-margin,
            ],
            out_std.prediction[
                :,
                pred_bands,
                margin:-margin,
                margin:-margin,
            ],
        )

        self.log(
            context + "/discriminator_dra",
            adv_loss_disc,
            batch_size=out_std.prediction.shape[0],
        )

        return self.config.adversarial.weight * adv_loss_gen

    def test_frequency_domain_analysis(
        self, batch: BatchData, predicted: PredictedData
    ):
        """
        Perform frequency domain analysis for validation purposes
        """
        # Compute bicubic up-sampled input tensor
        hr_scale_factor = (
            batch.target.shape[-1] / batch.network_input.hr_tensor.shape[-1]
        )
        input_tensor = torch.nn.functional.interpolate(
            batch.network_input.hr_tensor,
            scale_factor=hr_scale_factor,
            mode="bicubic",
            align_corners=False,
        )
        if batch.network_input.lr_tensor is not None:
            lr_scale_factor = (
                batch.target.shape[-1] / batch.network_input.lr_tensor.shape[-1]
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
        self.input_fft_profiles.append(profile)
        _, _, profile = compute_fft_profile(
            predicted.prediction, s=2 * input_tensor.shape[-1]
        )
        self.pred_fft_profiles.append(profile)
        _, _, profile = compute_fft_profile(batch.target, s=2 * input_tensor.shape[-1])
        self.target_fft_profiles.append(profile)
        self.fft_profiles_frequencies = freqs

    def validation_test_step(
        self,
        batch: BatchData,
        metrics: torch.nn.ModuleList,
        wald_metrics: torch.nn.ModuleList | None,
        real_losses: torch.nn.ModuleList | None,
        sim_losses: torch.nn.ModuleList | None,
        context: str = "validation",
    ) -> torch.Tensor | None:
        """
        Generic step for validation and test
        """
        # First convert batch to millirefl
        batch = batch_to_millirefl(batch, dtype=self.mean.dtype)

        # Align min / max per band on each patch
        if self.config.align_min_max:
            batch = align_min_max_batch(batch)

        # Derive simulated simulated simulated
        sim_batch = self.simulate_batch(batch)
        # Standardize both simulated and real batch
        batch_std = self.standardize_batch(batch)
        sim_batch_std = self.standardize_batch(sim_batch)

        if self.config.pretrain_registration:
            loss = self.registration_step(batch_std, context=context + "_losses")
            return loss
        # Forward pass on both batches
        with torch.no_grad():
            out_std = self.generator(batch_std.network_input)
            sim_out_std = self.generator(sim_batch_std.network_input)

        # Unstandardize predictions (for metrics)
        out_ustd = self.unstandardize_pred(out_std)
        sim_out_ustd = self.unstandardize_pred(sim_out_std)

        # Perform frequency domain analysis if in testing mode
        if context == "test":
            self.test_frequency_domain_analysis(batch, out_ustd)

        total_loss: torch.Tensor | None = None

        if self.config.registration and self.config.training_geometric_registration:
            batch_std_loss = self.register_target(
                batch_std, self.config.training_radiometric_registration
            )
        else:
            batch_std_loss = batch_std

        # Eval losses
        if real_losses is not None:
            # Evaluate pixel losses
            total_loss = self.eval_pixel_losses_and_metrics(
                batch_std_loss,
                out_std,
                real_losses,
                context=context + "_losses",
                margin=self.config.validation_margin,
                average=True,
            )
        if sim_losses is not None:
            sim_total_loss = self.eval_pixel_losses_and_metrics(
                sim_batch_std,
                sim_out_std,
                sim_losses,
                context=context + "_losses",
                margin=self.config.validation_margin,
                average=True,
            )
            if total_loss is not None:
                total_loss += sim_total_loss
            else:
                total_loss = sim_total_loss

        if self.config.registration and context == "test":
            hr_scale_factor = (
                batch_std.target.shape[-1] / batch_std.network_input.hr_tensor.shape[-1]
            )
            pred_registration_band = generic_downscale(
                out_std.prediction, factor=hr_scale_factor, mtf=0.3, padding="valid"
            )[:, self.config.registration.registration_channel, ...]
            target_registration_band = generic_downscale(
                batch_std.target, factor=hr_scale_factor, mtf=0.3, padding="valid"
            )[:, self.config.registration.registration_channel, ...]
            ref_registration_band = batch_std.network_input.hr_tensor[
                :, self.config.registration.registration_channel, ...
            ]
            flow = self.registration_module(
                pred_registration_band, ref_registration_band
            )
            target_flow = self.registration_module(
                target_registration_band, ref_registration_band
            )
            warp_loss = torch.norm(flow, p=2, dim=1).mean()
            std_warp_loss = torch.norm(flow, p=2, dim=1).std()
            target_warp_loss = torch.norm(target_flow, p=2, dim=1).mean()
            std_target_warp_loss = torch.norm(target_flow, p=2, dim=1).std()
            self.log(
                f"{context}_real_metrics/average_pred_warp",
                warp_loss,
                batch_size=batch_std.target.shape[0],
            )
            self.log(
                f"{context}_real_metrics/std_pred_warp",
                std_warp_loss,
                batch_size=batch_std.target.shape[0],
            )

            self.log(
                f"{context}_real_metrics/average_target_warp",
                target_warp_loss,
                batch_size=batch_std.target.shape[0],
            )

            self.log(
                f"{context}_real_metrics/std_target_warp",
                std_target_warp_loss,
                batch_size=batch_std.target.shape[0],
            )

        # Here we reset batch_std to be able to apply the testval
        # registration parameters Standardize both simulated and real
        # batch

        if self.config.registration and self.config.testval_geometric_registration:
            batch_std_metrics = self.register_target(
                batch_std, self.config.testval_radiometric_registration
            )
            batch.target = patches.unstandardize(
                batch_std_metrics.target,
                self.mean[:8],
                self.std[:8],
            )

        # Evaluate pixel metrics
        _ = self.eval_pixel_losses_and_metrics(
            batch,
            out_ustd,
            metrics,
            context=context + "_real_metrics",
            margin=self.config.validation_margin,
            average=True,
        )
        _ = self.eval_pixel_losses_and_metrics(
            sim_batch,
            sim_out_ustd,
            metrics,
            context=context + "_sim_metrics",
            margin=self.config.validation_margin,
            average=True,
        )
        # Evaluate Wald losses and metrics if required
        if wald_metrics is not None:
            wald_real_loss = self.wald_validation_test_step(
                batch, metrics=wald_metrics, context=context + "_real"
            )

            if total_loss is not None:
                total_loss += wald_real_loss
            else:
                total_loss = wald_real_loss

        # Evaluate adversarial losses
        if self.config.adversarial is not None:
            gen_adv_loss = self.adversarial_validation_test_step(
                batch_std, out_std, context=context + "_losses"
            )

            if total_loss is not None:
                total_loss += gen_adv_loss
            else:
                total_loss = gen_adv_loss

        if total_loss is not None:
            self.log(
                context + "_losses/total_loss",
                total_loss,
                batch_size=out_std.prediction.shape[0],
            )
        return total_loss

    def validation_step(self, batch: BatchData, _batch_idx: int):
        """
        Validation iteration
        """
        wald_metrics: torch.nn.ModuleList | None = None
        if self.config.wald is not None:
            wald_metrics = self.wald_validation_metrics

        self.validation_test_step(
            batch,
            metrics=self.validation_metrics,
            wald_metrics=wald_metrics,
            real_losses=self.real_losses,
            sim_losses=self.sim_losses,
            context="validation",
        )

    def test_step(self, batch: BatchData, _batch_idx: int):
        """
        Test iteration
        """
        wald_metrics: torch.nn.ModuleList | None = None
        if self.config.wald is not None:
            wald_metrics = self.wald_test_metrics

        self.validation_test_step(
            batch,
            metrics=self.test_metrics,
            wald_metrics=wald_metrics,
            real_losses=self.real_losses,
            sim_losses=self.sim_losses,
            context="test",
        )

    def on_test_epoch_end(self):
        """
        Synthetize frequency domain analysis
        """
        if self.config.pretrain_registration:
            return

        input_fft_profile = torch.cat(self.input_fft_profiles, dim=0).mean(dim=0)
        self.input_fft_profiles.clear()
        target_fft_profile = torch.cat(self.target_fft_profiles, dim=0).mean(dim=0)
        self.target_fft_profiles.clear()
        predicted_fft_profile = torch.cat(self.pred_fft_profiles, dim=0).mean(dim=0)
        self.pred_fft_profiles.clear()

        if (
            not input_fft_profile.shape[-1]
            == target_fft_profile.shape[-1]
            == predicted_fft_profile.shape[-1]
        ):

            logging.warning(
                "LR input and HR target have different number of \
                bands, cannot compute FDA metrics"
            )
            return

        fft_data = (
            torch.stack(
                (
                    input_fft_profile,
                    target_fft_profile,
                    predicted_fft_profile,
                ),
                dim=0,
            )
            .cpu()
            .numpy()
        )

        # Check current working directory
        assert self.logger
        assert self.logger.log_dir
        np.save(os.path.join(self.logger.log_dir, "fft_data.npy"), fft_data)
        np.save(
            os.path.join(self.logger.log_dir, "fft_freqs.npy"),
            self.fft_profiles_frequencies,
        )
        plot_fft_profile(
            fft_data,
            self.fft_profiles_frequencies.cpu().detach().numpy(),
            os.path.join(self.logger.log_dir, "fft_profiles.pdf"),
        )

        # Now compute fda metrics

        target_logprof = 10 * torch.log10(target_fft_profile) - 10 * torch.log10(
            target_fft_profile[1:2, ...]
        )

        input_logprof = 10 * torch.log10(input_fft_profile) - 10 * torch.log10(
            input_fft_profile[1:2, ...]
        )
        predicted_logprof = 10 * torch.log10(predicted_fft_profile) - 10 * torch.log10(
            predicted_fft_profile[1:2, ...]
        )

        for band_id, band in enumerate(self.config.standardization.bands[:8]):
            frr, arr, prr = compute_frr(
                predicted_logprof[:, band_id],
                target_logprof[:, band_id],
                input_logprof[:, band_id],
            )

            self.log(
                f"test/fda_prr_{band.value}",
                prr,
                on_epoch=True,
                on_step=False,
                batch_size=1,
            )

            self.log(
                f"test/fda_arr_{band.value}",
                arr,
                on_epoch=True,
                on_step=False,
                batch_size=1,
            )

            self.log(
                f"test/fda_frr_{band.value}",
                frr,
                on_epoch=True,
                on_step=False,
                batch_size=1,
            )
            fro, fru = compute_fro_fru(
                predicted_logprof[:, band_id],
                target_logprof[:, band_id],
                input_logprof[:, band_id],
            )

            self.log(
                f"test/fda_fro_{band.value}",
                fro,
                on_epoch=True,
                on_step=False,
                batch_size=1,
            )

            self.log(
                f"test/fda_fru_{band.value}",
                fru,
                on_epoch=True,
                on_step=False,
                batch_size=1,
            )
            fda_noise = compute_fda_noise(
                predicted_logprof[:, band_id],
                target_logprof[:, band_id],
                input_logprof[:, band_id],
            )

            self.log(
                f"test/fda_noise_{band.value}",
                fda_noise,
                on_epoch=True,
                on_step=False,
                batch_size=1,
            )

    def configure_optimizers(self):
        """
        Optimizer and scheduler
        """
        # Build main optimizer and scheduler
        if self.config.registration is not None and self.config.pretrain_registration:
            g_optimizer = torch.optim.Adam(
                chain(
                    self.generator.parameters(),
                    self.registration_module.parameters(),
                ),
                lr=self.config.optimization.learning_rate,
                betas=(0.9, 0.999),
            )
        else:
            g_optimizer = torch.optim.Adam(
                self.generator.parameters(),
                lr=self.config.optimization.learning_rate,
                betas=(0.9, 0.999),
            )
        optimizers = [g_optimizer]
        g_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            g_optimizer,
            T_0=self.config.optimization.t_0,
            T_mult=self.config.optimization.t_mult,
        )
        schedulers = [
            {
                "scheduler": g_scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]

        # Optionally build adversarial optimizers and schedulers
        if self.config.adversarial is not None and self.discriminator is not None:
            d_optimizer = torch.optim.Adam(
                params=self.discriminator.parameters(),
                lr=self.config.adversarial.optimization.learning_rate,
                betas=(0.9, 0.999),
            )
            optimizers.append(d_optimizer)
            d_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                d_optimizer,
                T_0=self.config.adversarial.optimization.t_0,
                T_mult=self.config.adversarial.optimization.t_mult,
            )
            schedulers.append(
                {
                    "scheduler": d_scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            )
        return optimizers, schedulers
