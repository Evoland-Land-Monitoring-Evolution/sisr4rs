# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Test code for the simulation of patches for the metrics study
"""
import torch

from torchsisr.simulation import simulate


def test_simulate() -> None:
    """
    Test the simulate routine
    """
    data = torch.zeros((1, 1, 32, 32))

    data[:, :, 16:, :] = 1.0
    _ = simulate(data)

    assert torch.all(data == simulate(data, None, None, None, None, None))

    test = simulate(data, None, 0.0, 0.0, 0.0, 0.0)

    assert torch.allclose(data, test)
