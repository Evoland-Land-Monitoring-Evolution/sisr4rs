#!/usr/bin/env python
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Tests for unet module
"""

import torch

from torchsisr.unet import UNet


def test_unet():
    """
    Test the Unet class
    """
    model = UNet(
        in_channels=3,
        out_channels=2,
        depth=3,
        min_skip_depth=1,
        up_mode="upsample",
        activation=torch.nn.functional.leaky_relu,
    )

    data = torch.rand((10, 3, 16, 16))

    out = model(data)
    # Trigger backprop
    out.sum().backward()

    for params in model.parameters():
        assert params.grad is not None
