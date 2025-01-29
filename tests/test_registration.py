#!/usr/bin/env python
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Tests for registration module
"""
import torch

from torchsisr.registration import UnetOpticalFlowEstimation, compose_flows, warp


def test_unet_optical_flow_estimation():
    """
    Test the UnetOpticalFlowEstimation class
    """

    model = UnetOpticalFlowEstimation()

    data1 = torch.rand((10, 16, 16))
    data2 = torch.rand((10, 16, 16))

    out = model(data1, data2)

    # Trigger backprop
    out.sum().backward()

    for params in model.parameters():
        assert params.grad is not None


def test_warp_function():
    """
    Test the warp function
    """
    data = torch.rand((10, 3, 16, 16))

    flow = torch.zeros((10, 2, 16, 16))

    warped = warp(data, flow)

    assert torch.allclose(data, warped)


def test_compose_flows_function():
    """
    Test the compose_flows function
    """
    shape = (10, 2, 16, 16)
    flow1 = torch.full(shape, 2.0)
    flow2 = torch.full(shape, 3.0)
    target_flow = torch.full(shape, 5.0)
    composed_flow = compose_flows(flow1, flow2)

    margin = 2

    assert torch.all(
        (composed_flow == target_flow)[:, :, margin:-margin, margin:-margin]
    )
