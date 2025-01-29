# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains types used in many places
"""
from dataclasses import dataclass
from typing import Self

import torch
from sensorsio.sentinel2 import Sentinel2
from torch import Tensor
from torch.nn import Module


@dataclass
class NetworkInput:
    """
    Represents model input
    """

    hr_tensor: Tensor
    hr_bands: tuple[Sentinel2.Band, ...]
    lr_tensor: None | Tensor = None
    lr_bands: tuple[Sentinel2.Band, ...] | None = None

    def __post_init__(self):
        """
        Post init asserts
        """
        assert isinstance(self.hr_bands, tuple)
        if len(self.hr_tensor.shape) == 4:
            assert self.hr_tensor.shape[1] == len(self.hr_bands)
        else:
            assert self.hr_tensor.shape[0] == len(self.hr_bands)
        assert (self.lr_tensor is self.lr_bands is None) or (
            self.lr_bands is not None and self.lr_tensor is not None
        )
        if self.lr_tensor is not None:
            assert isinstance(self.lr_bands, tuple)
            if len(self.lr_tensor.shape) == 4:
                assert self.lr_tensor.shape[1] == len(self.lr_bands)
            else:
                assert self.lr_tensor.shape[0] == len(self.lr_bands)


@dataclass
class BatchData:
    """
    Represent an element in batch
    """

    network_input: NetworkInput
    target: Tensor
    # which network ouptut channel correspond to which target channel
    target_bands: tuple[Sentinel2.Band, ...]

    def __post_init__(self):
        """
        Post init checks
        """
        assert isinstance(self.target_bands, tuple)
        if len(self.target.shape) == 4:
            assert self.target.shape[1] == len(self.target_bands)
        else:
            assert self.target.shape[0] == len(self.target_bands)

    def to(self, device=torch.device) -> Self:
        """
        Move batch to another device
        """
        self.network_input.hr_tensor = self.network_input.hr_tensor.to(device=device)

        if self.network_input.lr_tensor is not None:
            self.network_input.lr_tensor = self.network_input.lr_tensor.to(
                device=device
            )
        self.target = self.target.to(device=device)

        return self

    def pin_memory(self) -> Self:
        """
        Pin batch memory
        """
        self.network_input.hr_tensor = self.network_input.hr_tensor.pin_memory()
        if self.network_input.lr_tensor is not None:
            self.network_input.lr_tensor = self.network_input.lr_tensor.pin_memory()
        self.target = self.target.pin_memory()

        return self


@dataclass(frozen=True)
class PredictedData:
    """
    Represent network prediction
    """

    prediction: Tensor
    bands: tuple[Sentinel2.Band, ...]
    margin: int

    def __post_init__(self):
        """
        Post init checks
        """
        assert isinstance(self.bands, tuple)
        assert self.prediction.shape[1] == len(self.bands)
        assert 2 * self.margin < self.prediction.shape[2]
        assert 2 * self.margin < self.prediction.shape[3]


@dataclass(frozen=True)
class LossOutput:
    """
    This class represents a loss output
    """

    loss_values: Tensor
    bands: tuple[Sentinel2.Band, ...]

    def __post_init__(self):
        """
        Post init asserts
        """
        assert len(self.loss_values.shape) == 1
        assert self.loss_values.shape[0] == len(self.bands)


class ModelBase(Module):
    """
    Base class that add services
    """

    def get_prediction_margin(self) -> int:
        """
        Get the prediction margin
        """
        raise NotImplementedError()

    def get_upsampling_factor(self) -> float:
        """
        Returns the uspampling factor
        """
        raise NotImplementedError()
