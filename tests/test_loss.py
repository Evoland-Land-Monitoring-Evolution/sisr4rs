#!/usr/bin/env python
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
Tests for the loss.py module
"""

from collections import namedtuple

import pytest
import torch

from torchsisr import loss

ShiftLossConfig = namedtuple(
    "ShiftLossConfig",
    ["loss_fn", "optimize_min_max", "return_indices", "band20m"],
)


@pytest.fixture(
    name="shift_loss_factory",
    params=[
        ShiftLossConfig(loss.PerBandWrapper(), "min", True, False),
        ShiftLossConfig(loss.PerBandWrapper(), "min", False, False),
        ShiftLossConfig(
            loss.HRFidelity(factor=2.0, mtf=0.1),
            "min",
            False,
            False,
        ),
        ShiftLossConfig(
            loss.HRFidelity(factor=2.0, mtf=0.1),
            "min",
            True,
            True,
        ),
        ShiftLossConfig(
            loss.PerBandWrapper(loss.PeakSignalNoiseRatio()), "max", True, True
        ),
        ShiftLossConfig(
            loss.HRFidelity(factor=2.0, mtf=0.1),
            "min",
            False,
            True,
        ),
    ],
)
def fixture_model_factory(request):
    """
    Generic test function that handles different losses and metrics
    passed to shift wrapper
    """
    config = request.param
    pred = torch.full((10, 4, 16, 16), 0.5)
    ref = torch.full((10, 4, 16, 16), 0.5)
    ref[:, :, 0, :] = 0
    args = (pred, ref)
    return config, args


def test_per_band_wrapper_novgg():
    """
    Test the per band wrapper class
    """
    loss_fn = loss.PerBandWrapper()

    pred = torch.rand((10, 4, 16, 16))
    ref = torch.rand((10, 4, 16, 16))

    loss_value = loss_fn(pred, ref)

    for i in range(4):
        assert loss_value[i] == loss_fn.loss(pred[:, i, ...], ref[:, i, ...])


def test_lr_fidelity_loss():
    """
    Test the lr fidelity loss
    """
    loss_fn = loss.LRFidelity(factor=2.0, mtf=0.1)

    pred = torch.zeros((10, 4, 16, 16))
    ref = torch.zeros((10, 4, 8, 8))

    assert loss_fn(pred, ref)[0] == 0

    loss_fn = loss.LRFidelity(factor=4.0, mtf=0.1)
    pred = torch.zeros((10, 4, 32, 32))

    assert loss_fn(pred, ref)[0] == 0


def test_hr_fidelity_loss():
    """
    Test the hr fidelity loss
    """

    pred = torch.zeros((10, 4, 16, 16))
    ref = torch.zeros((10, 4, 16, 16))
    loss_fn = loss.HRFidelity(factor=2.0, mtf=0.1)
    assert loss_fn(pred, ref)[0] == 0
    loss_fn = loss.HRFidelity(factor=2.0, mtf=0.1)
    assert loss_fn(pred, ref)[0] == 0


def test_shift_wrapper(shift_loss_factory):
    """
    Test the shift wrapper class
    """
    config, args = shift_loss_factory
    loss_fn = config.loss_fn
    optimize_min_max = config.optimize_min_max
    return_indices = config.return_indices
    shift_value = 2
    minimizing = config.optimize_min_max == "min"
    wrapper = loss.ShiftWrapper(
        shift=shift_value,
        loss_fn=loss_fn,
        optimize_min_max=optimize_min_max,
        return_indices=return_indices,
    )
    loss_value = wrapper(*args)
    if return_indices:
        assert loss_value[0][0] == 0 if minimizing else torch.inf
        assert len(loss_value[1]) == args[0].shape[0]
    else:
        assert loss_value[0] == 0


def test_shift_results():
    """
    Test the results of shifts estimation
    """
    ref = torch.full((5, 4, 16, 16), 0.5)
    pred = torch.full((5, 4, 16, 16), 0.9)

    indices = torch.tensor([0, 12, 24, 3, 15])
    pred[0, :, 0:2, 0:2] = 0.5
    pred[1, :, 2:14, 2:14] = 0.5
    pred[2, :, 14:, 14:] = 0.5
    pred[3, :, :, 14] = 0.5
    pred[4, :, 14, :] = 0.5

    wrapper = loss.ShiftWrapper(
        shift=2,
        loss_fn=loss.PerBandWrapper(),
        optimize_min_max="min",
        return_indices=True,
    )

    best_loss, best_indices = wrapper(pred, ref)

    new_pred = torch.full((5, 4, 12, 12), 0.5)
    new_pred[0] = pred[0, :, 0:12, 0:12]
    new_pred[1] = pred[1, :, 2:14, 2:14]
    new_pred[2] = pred[2, :, 4:, 4:]
    new_pred[3] = pred[3, :, :12, 3:15]
    new_pred[4] = pred[4, :, 3:15, :12]

    smooth_loss = torch.nn.SmoothL1Loss()

    ref_loss = [
        smooth_loss(new_pred[:, b, :, :], ref[:, b, 2:14, 2:14])
        for b in range(ref.shape[1])
    ]
    ref_loss_stacked = torch.stack(ref_loss)

    assert torch.all(torch.eq(best_indices, indices))
    assert torch.all(
        torch.eq(
            torch.round(best_loss, decimals=5),
            torch.round(ref_loss_stacked, decimals=5),
        )
    )


def test_fft_hf_power_variation():
    """
    Test the FFT power variation loss
    """

    pred = torch.ones((10, 4, 16, 16))
    ref = torch.ones((10, 4, 16, 16))
    loss_fn = loss.FFTHFPowerVariation(scale_factor=1.0, support=0.25)

    assert loss_fn(pred, ref)[0] == 0

    ref = torch.ones((10, 4, 8, 8))
    loss_fn = loss.FFTHFPowerVariation(scale_factor=2.0, support=0.25)
    assert loss_fn(pred, ref)[0] == 0

    pred = torch.rand((10, 4, 16, 16))
    ref = torch.rand((10, 4, 8, 8))
    loss_fn(pred, ref)


def test_per_band_brisque():
    """
    Test the PerBandBRISQUE loss
    """
    pred = torch.rand((10, 4, 16, 16))
    ref = torch.rand((10, 4, 16, 16))

    per_band_brisque = loss.PerBandBRISQUE()

    assert torch.all(~torch.isnan(per_band_brisque(pred, ref)))


def test_per_band_brisque_variation():
    """
    Test the PerBandBRISQUEVariation loss
    """
    pred = torch.rand((10, 4, 16, 16))
    ref = torch.rand((10, 4, 16, 16))

    per_band_brisque = loss.PerBandBRISQUEVariation()
    assert torch.all(~torch.isnan(per_band_brisque(pred, ref)))


def test_per_band_tv_variation():
    """
    Test the PerBandTVVariation loss
    """
    pred = torch.rand((10, 4, 16, 16))
    ref = torch.rand((10, 4, 16, 16))

    per_band_tvv = loss.PerBandTVVariation()
    assert torch.all(~torch.isnan(per_band_tvv(pred, ref)))
