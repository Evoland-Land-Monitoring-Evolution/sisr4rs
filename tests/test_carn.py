#!/usr/bin/env python
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
"""
Tests for carn module
"""

import pytest
import torch

from torchsisr import carn


def test_eresblock():
    """
    Test output shape and gradient propagation for EResBlock class
    """
    nb_features = 16
    groups = 2
    kernel_size = 3
    w = h = 64

    in_tensor = torch.rand(10, nb_features, w, h)
    block = carn.EResBlock(nb_features, groups, kernel_size)
    out_tensor = block(in_tensor)

    # Test output shape
    assert out_tensor.shape, torch.Size([10, nb_features, w, h])

    # Trigger backprop
    out_tensor.sum().backward()

    total_params = 0
    for params in block.parameters():
        assert params.grad is not None
        total_params += params.flatten().shape[0]

    assert total_params == block.nb_params()


def test_cascading_block():
    """
    Test output shape and gradient propagation for cascading block class
    """
    nb_features = 16
    groups = 2
    kernel_size = 3
    w = h = 64

    in_tensor = torch.rand(10, nb_features, w, h)
    block = carn.CascadingBlock(
        3, nb_features, groups, kernel_size, shared_weights=True
    )
    block_nsw = carn.CascadingBlock(
        3, nb_features, groups, kernel_size, shared_weights=False
    )
    out_tensor = block(in_tensor)
    out_tensor_nsw = block_nsw(in_tensor)

    # Test output shape
    assert out_tensor.shape, torch.Size([10, nb_features, w, h])
    assert out_tensor_nsw.shape, torch.Size([10, nb_features, w, h])

    # Trigger backprop
    out_tensor.sum().backward()
    out_tensor_nsw.sum().backward()

    total_params = 0
    for params in block.parameters():
        assert params.grad is not None
        total_params += params.flatten().shape[0]

    total_params_nsw = 0
    for params in block_nsw.parameters():
        assert params.grad is not None
        total_params_nsw += params.flatten().shape[0]

    assert total_params == block.nb_params()
    assert total_params_nsw == block_nsw.nb_params()


@pytest.fixture(
    name="model_factory",
    params=[
        carn.CARNConfig(
            nb_bands=4,
            upsampling_factor=2.0,
            shared_weights=False,
            groups=1,
            nb_features_per_factor=2,
            nb_cascading_blocks=3,
            nb_eres_blocks_per_cascading_block=3,
        ),
        carn.CARNConfig(
            nb_bands=4,
            shared_weights=True,
            upsampling_factor=2.0,
            groups=1,
            nb_features_per_factor=2,
            nb_cascading_blocks=3,
            nb_eres_blocks_per_cascading_block=3,
        ),
        carn.CARNConfig(
            nb_bands=4,
            shared_weights=True,
            upsampling_factor=2.0,
            groups=2,
            nb_features_per_factor=2,
            nb_cascading_blocks=3,
            nb_eres_blocks_per_cascading_block=3,
        ),
        carn.CARNConfig(
            nb_bands=4,
            out_nb_bands=8,
            shared_weights=True,
            upsampling_factor=2.0,
            groups=2,
            nb_features_per_factor=3,
            nb_cascading_blocks=3,
            nb_eres_blocks_per_cascading_block=3,
        ),
    ],
)
def fixture_model_factory(request):
    """
    Generic configuration factory
    """
    config = request.param
    w = h = 64
    in_tensor = torch.rand(10, config.nb_bands, w, h)
    model = carn.CARN(config)
    out_tensor = model.predict(in_tensor)
    out_tensor = model(in_tensor)

    return config, model, out_tensor


def test_carn_model_arch(model_factory):
    """
    Test the SuperResolutionModel output shape
    """
    config, model, _ = model_factory

    assert len(model.cascading_blocks) == config.nb_cascading_blocks

    for b in model.cascading_blocks:
        assert len(b.residual_blocks) == config.nb_eres_blocks_per_cascading_block

    total_params = 0
    for params in model.parameters():
        total_params += params.flatten().shape[0]

    assert total_params == model.nb_params()


def test_carn_model_outshape(model_factory):
    """
    Test the SuperResolutionModel output shape
    """
    config, _, out_tensor = model_factory
    if config.out_nb_bands is None:
        assert out_tensor.shape == torch.Size([10, config.nb_bands, 128, 128])
    else:
        assert out_tensor.shape == torch.Size([10, config.out_nb_bands, 128, 128])


def test_carn_model_backprop(model_factory):
    """
    Test the SuperResolutionModel gradient backpropagation
    """
    _, model, out_tensor = model_factory
    out_tensor.sum().backward()

    print(model)

    for c_i, c_block in enumerate(model.cascading_blocks):
        for params in c_block.parameters():
            assert params.grad is not None, f"cascading: {c_i}"
        for r_i, r_block in enumerate(c_block.residual_blocks):
            for params in r_block.parameters():
                assert params.grad is not None, f"cascading: {c_i}, eres : {r_i}"
    for r_i, r_block in enumerate(model.residual_1d_conv):
        for params in r_block.parameters():
            assert params.grad is not None

    for params in model.input_conv.parameters():
        assert params is not None

    for params in model.output_conv1.parameters():
        assert params is not None

    for params in model.output_conv2.parameters():
        assert params is not None

    for params in model.parameters():
        print(params.shape)
        assert params.grad is not None
