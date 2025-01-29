# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the tests from simplified_training_module.py
"""
import pytest
import pytorch_lightning as pl
import torch
from sensorsio.sentinel2 import Sentinel2 as s2

from torchsisr import carn, double_sisr_model, training
from torchsisr.custom_types import BatchData, NetworkInput
from torchsisr.dataset import batch_data_collate_fn
from torchsisr.discriminator import Discriminator
from torchsisr.loss import (
    GradientStrataWrapper,
    HRFidelity,
    LRFidelity,
    PeakSignalNoiseRatio,
    PerBandWrapper,
    RMSELoss,
)
from torchsisr.loss_helper import (
    AgainstHRInputPixelLossWrapper,
    AgainstLRInputPixelLossWrapper,
    PixelLossWrapper,
)


class FakeSen2VnsDataset(torch.utils.data.Dataset):
    """
    Generate fake data
    """

    def __init__(
        self,
        nb_patches: int = 1000,
        hr_patch_width: int = 128,
        generate_20m_data: bool = True,
        generate_b11b12_data: bool = True,
    ):
        """
        Class Constructor
        """
        super().__init__()

        self.nb_patches = nb_patches
        self.hr_patch_width = hr_patch_width
        self.generate_20m_data = generate_20m_data
        self.generate_b11b12_data = generate_b11b12_data

    def __len__(self):
        """ """
        return self.nb_patches

    def __getitem__(self, idx: int) -> BatchData:
        """ """
        assert idx < self.nb_patches

        # Generate 10m data
        hr_tensor = 5000 + 10000 * torch.rand(
            (4, self.hr_patch_width, self.hr_patch_width)
        )
        hr_bands = tuple(s2.GROUP_10M)

        lr_tensor: torch.Tensor | None = None
        lr_bands: tuple[s2.Band, ...] | None = None

        if self.generate_20m_data:
            target_tensor = 5000 + 10000 * torch.rand(
                (8, 2 * self.hr_patch_width, 2 * self.hr_patch_width)
            )
            target_bands = tuple(s2.GROUP_10M + s2.GROUP_20M[:-2])
            width = int(self.hr_patch_width / 2)
            if self.generate_b11b12_data:
                lr_tensor = 5000 + 10000 * torch.rand((6, width, width))
                lr_bands = tuple(s2.GROUP_20M)
            else:
                lr_tensor = 5000 + 10000 * torch.rand((4, width, width))
                lr_bands = tuple(s2.GROUP_20M[:-2])
        else:
            target_tensor = 5000 + 10000 * torch.rand(
                (4, 2 * self.hr_patch_width, 2 * self.hr_patch_width)
            )
            target_bands = tuple(s2.GROUP_10M)

        network_input = NetworkInput(hr_tensor, hr_bands, lr_tensor, lr_bands)

        return BatchData(network_input, target=target_tensor, target_bands=target_bands)


class FakeSen2VnsDataModule(pl.LightningDataModule):
    """
    A fake sen2vns datamodule designed for tests
    """

    def __init__(
        self,
        nb_training_patches: int = 20,
        nb_validation_patches: int = 10,
        nb_testing_patches: int = 10,
        hr_patch_width: int = 128,
        generate_20m_data: bool = True,
        generate_b11b12_data: bool = True,
        batch_size: int = 2,
    ):
        """
        Class Constructor
        """
        super().__init__()

        self.batch_size = batch_size

        self.training_dataset = FakeSen2VnsDataset(
            nb_training_patches, hr_patch_width, generate_20m_data, generate_b11b12_data
        )

        self.validation_dataset = FakeSen2VnsDataset(
            nb_validation_patches,
            hr_patch_width,
            generate_20m_data,
            generate_b11b12_data,
        )
        self.testing_dataset = FakeSen2VnsDataset(
            nb_testing_patches, hr_patch_width, generate_20m_data, generate_b11b12_data
        )

    def train_dataloader(self):
        """
        Return train dataloaded (reset every time this method is called)
        """
        return torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=1,
            collate_fn=batch_data_collate_fn,
            prefetch_factor=4,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Return validation data loader (never reset)
        """
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=1,
            collate_fn=batch_data_collate_fn,
            prefetch_factor=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        Return test data loader (never reset)
        """
        return torch.utils.data.DataLoader(
            self.testing_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=False,
            collate_fn=batch_data_collate_fn,
            num_workers=1,
            pin_memory=True,
        )


def test_fake_dataset():
    """
    Test the fake sen2vns dataset
    """

    fake_dataset = FakeSen2VnsDataset()

    batch = fake_dataset[10]

    assert batch.network_input.lr_bands
    assert len(batch.network_input.lr_bands) == 6
    assert len(batch.network_input.hr_bands) == 4


def generate_config():
    """
    Generate configuration for DoubleSISRTrainingModule
    """
    bands = (s2.B2, s2.B3, s2.B4, s2.B5, s2.B6, s2.B7, s2.B8, s2.B8A, s2.B11, s2.B12)

    # Build standardization parameters
    std_params = training.StandardizationParameters(
        bands=bands,
        mean=tuple(0.0 for i in range(10)),
        std=tuple(1.0 for i in range(10)),
    )

    # Build batch simulation parameters
    batch_sim_params = training.BatchSimulationParameters(
        mtf_min=0.1,
        mtf_max=0.2,
        noise_multiplier_min=0.9,
        noise_multiplier_max=1.1,
        noise_std=tuple(0.0 for i in range(8)),
    )

    # Build super-resolution model
    carn_config = carn.CARNConfig(
        nb_bands=10,
        upsampling_factor=2.0,
        nb_cascading_blocks=1,
        nb_features_per_factor=2,
    )
    carn_model = carn.CARN(carn_config)
    double_carn_model = double_sisr_model.DoubleSuperResolutionModel(carn_model)

    # Build scheduler and optimizer
    optimization_params = training.OptimizationParameters(
        learning_rate=0.0001, t_0=750, t_mult=2
    )

    # Build adversarial parameters
    discriminator = Discriminator(in_features=3, high_pass_filtering_mtf=0.4)

    # Build scheduler and optimizer
    d_optimization_params = training.OptimizationParameters(
        learning_rate=0.00001, t_0=750, t_mult=2
    )

    adversarial_parameters = training.AdversarialParameters(
        discriminator=discriminator,
        optimization=d_optimization_params,
        bands=(s2.B2, s2.B3, s2.B4),
        weight=0.1,
        mode="BCE",
        real_label_smoothing=0.1,
        starting_step_discriminator=1,
        starting_step_generator=1,
    )
    # Build losses
    sim_losses = (
        PixelLossWrapper(
            HRFidelity(
                loss=torch.nn.SmoothL1Loss(),
                factor=2.0,
                mtf=0.1,
            ),
            bands=(s2.B2, s2.B3, s2.B4, s2.B8),
            name="hr_l1",
        ),
        PixelLossWrapper(
            HRFidelity(
                loss=torch.nn.SmoothL1Loss(),
                factor=4.0,
                mtf=0.1,
            ),
            bands=(s2.B5, s2.B6, s2.B7, s2.B8A),
            name="hr_l1",
        ),
    )
    real_losses = (
        AgainstHRInputPixelLossWrapper(
            LRFidelity(
                loss=torch.nn.SmoothL1Loss(),
                factor=2.0,
                mtf=0.1,
            ),
            bands=(s2.B2, s2.B3, s2.B4, s2.B8),
            name="lr_l1",
        ),
        AgainstLRInputPixelLossWrapper(
            LRFidelity(loss=torch.nn.SmoothL1Loss(), factor=4.0, mtf=0.1),
            bands=(s2.B5, s2.B6, s2.B7, s2.B8A),
            name="lr_l1",
        ),
    )
    # Build validation metrics
    validation_metrics = (
        PixelLossWrapper(
            PerBandWrapper(loss=PeakSignalNoiseRatio(data_range=1.0)),
            bands=bands[:-2],
            name="psnr",
        ),
        AgainstHRInputPixelLossWrapper(
            LRFidelity(loss=RMSELoss(), factor=2.0, mtf=0.1),
            bands=(s2.B2, s2.B3, s2.B4, s2.B8),
            name="lr_rmse",
        ),
        AgainstLRInputPixelLossWrapper(
            LRFidelity(loss=RMSELoss(), factor=4.0, mtf=0.1),
            bands=(s2.B5, s2.B6, s2.B7, s2.B8A),
            name="lr_rmse",
        ),
        PixelLossWrapper(
            HRFidelity(
                loss=RMSELoss(),
                factor=2.0,
                mtf=0.1,
            ),
            bands=(s2.B2, s2.B3, s2.B4, s2.B8),
            name="hr_rmse",
        ),
        PixelLossWrapper(
            HRFidelity(
                loss=RMSELoss(),
                factor=4.0,
                mtf=0.1,
            ),
            bands=(s2.B5, s2.B6, s2.B7, s2.B8A),
            name="hr_rmse",
        ),
        PixelLossWrapper(
            GradientStrataWrapper(
                loss=RMSELoss(),
                grad_mag_min=0.006770164705812931,
                grad_mag_max=1.7976931348623157e308,
            ),
            name="high_grad_strata_rmse",
            bands=bands[:-2],
        ),
    )

    # Build test metrics
    test_metrics = validation_metrics

    # Build Wald losses
    wald_losses = (
        PixelLossWrapper(
            HRFidelity(loss=torch.nn.SmoothL1Loss(), factor=4.0, mtf=0.1),
            bands=(s2.B11, s2.B12),
            name="hr_l1",
        ),
        AgainstLRInputPixelLossWrapper(
            LRFidelity(loss=torch.nn.SmoothL1Loss(), factor=4.0, mtf=0.1),
            bands=(s2.B11, s2.B12),
            name="lr_l1",
        ),
    )
    # Build Wald metrics
    wald_validation_metrics = (
        PixelLossWrapper(
            PerBandWrapper(loss=PeakSignalNoiseRatio(data_range=1.0)),
            bands=bands[:-2],
            name="psnr",
        ),
        AgainstLRInputPixelLossWrapper(
            LRFidelity(
                loss=RMSELoss(),
                factor=4.0,
                mtf=0.1,
            ),
            bands=(s2.B11, s2.B12),
            name="lr_rmse",
        ),
        PixelLossWrapper(
            HRFidelity(loss=RMSELoss(), factor=4.0, mtf=0.1),
            bands=(s2.B11, s2.B12),
            name="hr_rmse",
        ),
    )

    # Build Wald test metrics
    wald_test_metrics = wald_validation_metrics

    wald_parameters = training.WaldParameters(
        losses=wald_losses,
        validation_metrics=wald_validation_metrics,
        test_metrics=wald_test_metrics,
        validation_margin=16,
        noise_multiplier_min=0.9,
        noise_multiplier_max=1.1,
        mtf_min=0.1,
        mtf_max=0.2,
        noise_std=tuple(0.01 for b in range(10)),
        pad_to_input_size=True,
    )
    # Build final config
    config = training.DoubleSISRTrainingModuleConfig(
        model=double_carn_model,
        optimization=optimization_params,
        adversarial=adversarial_parameters,
        wald=wald_parameters,
        standardization=std_params,
        batch_simulation=batch_sim_params,
        real_losses=real_losses,
        sim_losses=sim_losses,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_margin=32,
        align_min_max=True,
    )

    return config


def test_training_class_instantiation():
    """
    Test training class instantiation
    """
    config = generate_config()

    training.DoubleSISRTrainingModule(config)


def test_training_class_fit_cpu():
    """
    Test training class fit
    """
    config = generate_config()

    model = training.DoubleSISRTrainingModule(config)

    dm = FakeSen2VnsDataModule(batch_size=10)

    model_summary_cb = pl.callbacks.RichModelSummary(max_depth=-1)
    progress_bar_cb = pl.callbacks.RichProgressBar(leave=True)

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        callbacks=[model_summary_cb, progress_bar_cb],
        log_every_n_steps=1,
        accelerator="cpu",
    )

    trainer.fit(model, dm)


def test_training_class_validate_cpu():
    """
    Test training class test
    """
    config = generate_config()

    model = training.DoubleSISRTrainingModule(config)

    dm = FakeSen2VnsDataModule(batch_size=2)

    trainer = pl.Trainer(
        max_epochs=1, limit_val_batches=2, log_every_n_steps=1, accelerator="cpu"
    )

    trainer.validate(model, dm)


def test_training_class_test_cpu():
    """
    Test training class test
    """
    config = generate_config()

    model = training.DoubleSISRTrainingModule(config)

    dm = FakeSen2VnsDataModule(batch_size=2)

    trainer = pl.Trainer(
        max_epochs=1, limit_test_batches=2, log_every_n_steps=1, accelerator="cpu"
    )

    trainer.test(model, dm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_training_class_fit_gpu():
    """
    Test training class fit
    """
    config = generate_config()

    model = training.DoubleSISRTrainingModule(config)

    dm = FakeSen2VnsDataModule(batch_size=2)

    model_summary_cb = pl.callbacks.RichModelSummary(max_depth=-1)
    progress_bar_cb = pl.callbacks.RichProgressBar(leave=True)

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        callbacks=[model_summary_cb, progress_bar_cb],
        log_every_n_steps=1,
        accelerator="gpu",
    )

    trainer.fit(model, dm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_training_class_fit_gpu_bf16():
    """
    Test training class fit
    """
    config = generate_config()
    model = training.DoubleSISRTrainingModule(config)

    dm = FakeSen2VnsDataModule(batch_size=2)

    model_summary_cb = pl.callbacks.RichModelSummary(max_depth=-1)
    progress_bar_cb = pl.callbacks.RichProgressBar(leave=True)

    trainer = pl.Trainer(
        max_epochs=1,
        precision="bf16",
        val_check_interval=0.5,
        callbacks=[model_summary_cb, progress_bar_cb],
        log_every_n_steps=1,
        accelerator="gpu",
    )
    trainer.fit(model, dm)
