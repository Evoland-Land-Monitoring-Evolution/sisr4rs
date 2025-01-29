#!/usr/bin/env python
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the SuperResolutionModel, that allows to
handle jointly 10m and 20m bands from Sentinel2.
"""

import torch

from torchsisr.custom_types import ModelBase, NetworkInput, PredictedData


class BicubicInterpolation(ModelBase):
    """
    This class upscale 20m bands to 10m by using bicubic interpolation,
    if selected by user

    Constructor:
    : param upsampling_factor: We upscale the input image that much times
    """

    def __init__(self, upsampling_factor: float = 2.0):
        super().__init__()
        self.upsampling_factor = upsampling_factor

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Bicubic interpolation branch
        :param data: Input tensor of shape [nb_samples,nb_features,width,height]
        :return: Output tensor of shape [nb_samples,nb_features,
        upsampling_factor*width, upsampling_factor*height]
        """
        return torch.nn.functional.interpolate(
            data,
            scale_factor=self.upsampling_factor,
            align_corners=False,
            mode="bicubic",
        )

    def get_prediction_margin(self) -> int:
        """
        Prediction margin is 1
        """
        return 1

    def get_upsampling_factor(self) -> float:
        """
        Prediction margin is 1
        """
        return self.upsampling_factor


class DoubleSuperResolutionModel(torch.nn.Module):
    """
    Super resolution model that allows to handle:
    - Single band group upsampling (either 10m or 20m)
    - Joint upsampling of 10m and 20m bands)
    """

    def __init__(
        self,
        sisr_model: ModelBase,
        lr_to_hr_model: None | ModelBase = BicubicInterpolation(),
    ):
        """
        Constructor:
        :param sisr_module: The super-resolution module
        :param lr_to_hr_module: The optional module to bring
        lr to hr input resolution

        """
        super().__init__()

        self.sisr_model = sisr_model
        self.lr_to_hr_model = lr_to_hr_model

    def forward(self, data: NetworkInput) -> PredictedData:
        """
        Forward pass of Super Resolution model

        :param data: Network input
        :return: Output tensor
        """

        hr_input = data.hr_tensor
        predicted_bands = data.hr_bands

        if self.lr_to_hr_model is not None:
            assert data.lr_tensor is not None
            assert data.lr_bands is not None
            lr_to_hr_tensor = self.lr_to_hr_model(data.lr_tensor)
            hr_input = torch.cat((hr_input, lr_to_hr_tensor), dim=1)
            predicted_bands = tuple(
                list(predicted_bands) + list(data.lr_bands)
            )  # Ugly way to concatenate tuples
        return PredictedData(
            self.sisr_model(hr_input), predicted_bands, self.get_prediction_margin()
        )

    def predict(self, data: NetworkInput) -> PredictedData:
        """
        Same as forward method but with final unstardardization and no_grad

        :param data:
        :return: Output tensor
        """
        with torch.no_grad():
            return self.forward(data)

    def get_prediction_margin(self) -> int:
        """
        This code compute the margin added to the image patches during
        the prediction step. Predicted margin pixels will have
        the "border effect". So when we reconstruct the image from predicted
        patches, the margins are not taken into account.
        """
        if self.lr_to_hr_model:
            return int(
                self.sisr_model.get_prediction_margin()
                + self.sisr_model.get_upsampling_factor()
                * self.lr_to_hr_model.get_prediction_margin()
            )
        return self.sisr_model.get_prediction_margin()
