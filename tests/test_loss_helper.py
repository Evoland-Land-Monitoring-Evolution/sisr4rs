#!/usr/bin/env python
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
Tests for the loss_helper.py module
"""
import torch
from sensorsio import sentinel2

from torchsisr import loss_helper
from torchsisr.custom_types import BatchData, NetworkInput, PredictedData
from torchsisr.loss import HRFidelity, LRFidelity, PerBandWrapper, ShiftWrapper


def build_data():
    """
    Prepare data for test
    """
    network_input = NetworkInput(
        hr_tensor=torch.rand((9, 4, 16, 16)),
        hr_bands=tuple(sentinel2.Sentinel2.GROUP_10M),
        lr_tensor=torch.rand((9, 6, 8, 8)),
        lr_bands=tuple(sentinel2.Sentinel2.GROUP_20M),
    )
    batch_data = BatchData(
        network_input,
        target=torch.rand((9, 8, 32, 32)),
        target_bands=tuple(
            sentinel2.Sentinel2.GROUP_10M + sentinel2.Sentinel2.GROUP_20M[:-2]
        ),
    )

    pred_data = PredictedData(
        torch.rand((9, 10, 32, 32)),
        bands=tuple(sentinel2.Sentinel2.GROUP_10M + sentinel2.Sentinel2.GROUP_20M),
        margin=12,
    )
    return batch_data, pred_data


def test_pixel_loss_helper():
    """
    test the PixelLossWrapper class
    """
    loss = PerBandWrapper(torch.nn.SmoothL1Loss())

    loss_wrapper = loss_helper.PixelLossWrapper(
        loss=loss,
        name="smooth_l1",
        weight=0.5,
        bands=tuple(sentinel2.Sentinel2.GROUP_10M + sentinel2.Sentinel2.GROUP_20M[:-2]),
    )

    batch_data, pred_data = build_data()

    loss_output = loss_wrapper.forward(batch_data, pred_data)

    assert loss_output.bands == tuple(
        sentinel2.Sentinel2.GROUP_10M + sentinel2.Sentinel2.GROUP_20M[:-2]
    )


def test_hr_fidelity_hr_input_pixel_loss_helper():
    """
    test the PixelLossWrapper class
    """
    loss = HRFidelity(factor=2.0, mtf=0.1)

    loss_wrapper = loss_helper.PixelLossWrapper(
        loss=loss,
        name="smooth_l1",
        weight=0.5,
        bands=tuple(sentinel2.Sentinel2.GROUP_10M),
    )

    batch_data, pred_data = build_data()

    loss_output = loss_wrapper.forward(batch_data, pred_data)

    assert loss_output.bands == tuple(sentinel2.Sentinel2.GROUP_10M)


def test_hr_fidelity_lr_input_pixel_loss_helper():
    """
    test the PixelLossWrapper class
    """
    loss = HRFidelity(factor=4.0, mtf=0.1)

    loss_wrapper = loss_helper.PixelLossWrapper(
        loss=loss,
        name="smooth_l1",
        weight=0.5,
        bands=tuple(sentinel2.Sentinel2.GROUP_20M[:-2]),
    )

    batch_data, pred_data = build_data()

    loss_output = loss_wrapper.forward(batch_data, pred_data)

    assert loss_output.bands == tuple(sentinel2.Sentinel2.GROUP_20M[:-2])


def test_pixel_loss_helper_shift_wrapper():
    """
    test the PixelLossWrapper class
    """
    # For now, we are not compatible with return_indices=True
    loss = ShiftWrapper(
        loss_fn=PerBandWrapper(torch.nn.SmoothL1Loss()), return_indices=False
    )

    loss_wrapper = loss_helper.PixelLossWrapper(
        loss=loss,
        name="smooth_l1",
        weight=0.5,
        bands=tuple(sentinel2.Sentinel2.GROUP_10M + sentinel2.Sentinel2.GROUP_20M[:-2]),
    )

    batch_data, pred_data = build_data()

    loss_output = loss_wrapper.forward(batch_data, pred_data)

    assert loss_output.bands == tuple(
        sentinel2.Sentinel2.GROUP_10M + sentinel2.Sentinel2.GROUP_20M[:-2]
    )


def test_lr_fidelity_hr_input_pixel_loss_helper():
    """
    test the PixelLossWrapper class
    """
    loss = LRFidelity(factor=2, mtf=0.1)

    loss_wrapper = loss_helper.AgainstHRInputPixelLossWrapper(
        loss=loss,
        name="lr_fidelity",
        weight=0.5,
        bands=tuple(sentinel2.Sentinel2.GROUP_10M),
    )
    batch_data, pred_data = build_data()

    loss_output = loss_wrapper(batch_data, pred_data)

    assert loss_output.bands == tuple(sentinel2.Sentinel2.GROUP_10M)


def test_lr_fidelity_lr_input_pixel_loss_helper():
    """
    test the PixelLossWrapper class
    """
    loss = LRFidelity(factor=4.0, mtf=None)

    loss_wrapper = loss_helper.AgainstLRInputPixelLossWrapper(
        loss=loss,
        name="lr_fidelity",
        weight=0.5,
        bands=tuple(sentinel2.Sentinel2.GROUP_20M),
    )
    batch_data, pred_data = build_data()

    loss_output = loss_wrapper(batch_data, pred_data)

    assert loss_output.bands == tuple(sentinel2.Sentinel2.GROUP_20M)
