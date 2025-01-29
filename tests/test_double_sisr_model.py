#!/usr/bin/env python
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
Test the double_carn module
"""

from dataclasses import dataclass

import pytest
import torch
from sensorsio.sentinel2 import Sentinel2

from torchsisr import carn
from torchsisr import custom_types as types
from torchsisr import double_sisr_model


def test_bicubic_interpolation():
    """
    Test the bicubic interpolation module
    """
    nb_bands = 4
    upsampling_factor = 2
    w = h = 64

    in_tensor = torch.rand(10, nb_bands, w, h)
    model = double_sisr_model.BicubicInterpolation(upsampling_factor)
    out_tensor = model(in_tensor)

    assert out_tensor.shape == torch.Size(
        [10, nb_bands, w * upsampling_factor, h * upsampling_factor]
    )
    assert model.get_prediction_margin() == 1


# Named tuple for different double carn config
@dataclass(frozen=True)
class DoubleConfig:
    """
    Configuration for DoubleSISR class
    """

    double: bool
    use_interp: bool
    nb_hr_bands: int
    hr_sisr_factor: float
    nb_lr_bands: int | None = None
    lr_to_hr_factor: float | None = None


@pytest.fixture(
    name="model_factory",
    params=[
        DoubleConfig(False, False, nb_hr_bands=4, hr_sisr_factor=2.0),
        DoubleConfig(
            True,
            False,
            nb_hr_bands=4,
            nb_lr_bands=6,
            hr_sisr_factor=2.0,
            lr_to_hr_factor=3.0,
        ),
        DoubleConfig(
            True,
            True,
            nb_hr_bands=4,
            nb_lr_bands=6,
            hr_sisr_factor=2.0,
            lr_to_hr_factor=3.0,
        ),
    ],
)
def fixture_model_factory(
    request,
) -> tuple[
    DoubleConfig,
    double_sisr_model.DoubleSuperResolutionModel,
    types.NetworkInput,
    torch.Tensor,
]:
    """
    Generic test function that handles both the
    interp and double carn cases
    """
    config = request.param

    sisr_model = carn.CARN(
        carn.CARNConfig(
            nb_bands=(
                config.nb_hr_bands + config.nb_lr_bands
                if config.double
                else config.nb_hr_bands
            ),
            upsampling_factor=config.hr_sisr_factor,
            nb_cascading_blocks=1,
        )
    )
    hr_to_lr_model: types.ModelBase | None = None

    if config.double:
        if config.use_interp:
            hr_to_lr_model = double_sisr_model.BicubicInterpolation(
                upsampling_factor=config.lr_to_hr_factor
            )
        else:
            hr_to_lr_model = carn.CARN(
                carn.CARNConfig(
                    nb_bands=config.nb_lr_bands,
                    upsampling_factor=config.lr_to_hr_factor,
                    nb_cascading_blocks=1,
                )
            )

    model = double_sisr_model.DoubleSuperResolutionModel(
        sisr_model=sisr_model, lr_to_hr_model=hr_to_lr_model
    )
    batch_size = 10
    patch_size_lr = 32
    if config.double:
        patch_size_hr = int(config.lr_to_hr_factor * patch_size_lr)
    else:
        patch_size_hr = patch_size_lr

    in_data = types.NetworkInput(
        torch.rand(batch_size, config.nb_hr_bands, patch_size_hr, patch_size_hr),
        (Sentinel2.B2, Sentinel2.B3, Sentinel2.B4, Sentinel2.B8),
    )
    if config.double:
        in_data = types.NetworkInput(
            torch.rand(batch_size, config.nb_hr_bands, patch_size_hr, patch_size_hr),
            (Sentinel2.B2, Sentinel2.B3, Sentinel2.B4, Sentinel2.B8),
            torch.rand(batch_size, config.nb_lr_bands, patch_size_lr, patch_size_lr),
            (
                Sentinel2.B5,
                Sentinel2.B6,
                Sentinel2.B7,
                Sentinel2.B8A,
                Sentinel2.B11,
                Sentinel2.B12,
            ),
        )

    out_tensor = model(in_data).prediction
    _ = model.predict(in_data)
    return request.param, model, in_data, out_tensor


def test_super_resolution_model_outshape(model_factory) -> None:
    """
    Test the SuperResolutionModel output shape
    """
    config, _, in_data, out_tensor = model_factory
    nb_expected_bands = (
        config.nb_hr_bands + config.nb_lr_bands if config.double else config.nb_hr_bands
    )
    expected_patch_size = int(config.hr_sisr_factor * in_data.hr_tensor.shape[2])
    expected_shape = torch.Size(
        (
            in_data.hr_tensor.shape[0],
            nb_expected_bands,
            expected_patch_size,
            expected_patch_size,
        )
    )

    assert out_tensor.shape == expected_shape


def test_super_resolution_model_backprop(model_factory) -> None:
    """
    Test the SuperResolutionModel gradient backpropagation
    """
    _, model, _, out_tensor = model_factory
    out_tensor.sum().backward()
    for params in model.parameters():
        assert params.grad is not None


def test_super_resolution_model_compute_margin(model_factory) -> None:
    """
    Test the computation of prediction margins
    """
    _, model, _, _ = model_factory

    _ = model.get_prediction_margin()
