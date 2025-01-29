# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains loss helper classes that simplifies the routing
of losses input in training module

"""
import numpy as np
import torch
from sensorsio import sentinel2

from torchsisr.custom_types import BatchData, LossOutput, PredictedData
from torchsisr.dataset import match_bands


class PixelLossWrapper(torch.nn.Module):
    """
    Wrapper class for all pixel losses
    """

    def __init__(
        self,
        loss: torch.nn.Module,
        name: str,
        bands: tuple[sentinel2.Sentinel2.Band, ...],
        weight: float = 1.0,
    ):
        """
        :param loss: the loss function
        :param name: the name of the loss
        :param weight: the weight of the loss
        """
        super().__init__()
        self.name = name
        self.loss = loss
        self.weight = weight
        self.bands = bands

        if isinstance(bands[0], str):
            self.bands = tuple(sentinel2.Sentinel2.Band(b) for b in bands)
        else:
            self.bands = bands

    def forward(
        self, batch: BatchData, pred: PredictedData, margin: int | None = None
    ) -> LossOutput:
        """
        Default forward against pred
        """
        pred_bands, _ = match_bands(self.bands, pred.bands)
        assert len(pred_bands) == len(self.bands)
        target_bands, _ = match_bands(self.bands, batch.target_bands)
        assert len(target_bands) == len(self.bands)

        if margin is None:
            actual_margin = pred.margin
        else:
            actual_margin = margin

        loss_value = self.weight * self.loss(
            pred.prediction[
                :,
                pred_bands,
                actual_margin:-actual_margin,
                actual_margin:-actual_margin,
            ],
            batch.target[
                :,
                target_bands,
                actual_margin:-actual_margin,
                actual_margin:-actual_margin,
            ],
        )
        # Add required dim if missing
        if len(loss_value.shape) == 0:
            loss_value = loss_value[None, ...]

        return LossOutput(
            loss_value,
            self.bands,
        )


class AgainstHRInputPixelLossWrapper(PixelLossWrapper):
    """
    Subclass of pixel loss that evaluates against hr input
    """

    def forward(
        self, batch: BatchData, pred: PredictedData, margin: int | None = None
    ) -> LossOutput:
        """
        Evaluate against hr inputs
        """
        pred_bands, _ = match_bands(self.bands, pred.bands)
        assert len(pred_bands) == len(self.bands)
        target_bands, _ = match_bands(self.bands, batch.network_input.hr_bands)
        assert len(target_bands) == len(self.bands)

        if margin is None:
            actual_margin = pred.margin
        else:
            actual_margin = margin

        factor = (
            float(pred.prediction.shape[-1]) / batch.network_input.hr_tensor.shape[-1]
        )

        # Make sure actual margin can be divided by factor
        actual_margin = int(factor * np.ceil(actual_margin / factor))

        actual_lr_margin = int(actual_margin / factor)
        hr_loss = self.weight * self.loss(
            pred.prediction[
                :,
                pred_bands,
                actual_margin:-actual_margin,
                actual_margin:-actual_margin,
            ],
            batch.network_input.hr_tensor[
                :,
                target_bands,
                actual_lr_margin:-actual_lr_margin,
                actual_lr_margin:-actual_lr_margin,
            ],
        )

        return LossOutput(self.weight * hr_loss, self.bands)


class AgainstLRInputPixelLossWrapper(PixelLossWrapper):
    """
    Subclass of pixel loss that evaluates against hr input
    """

    def forward(
        self, batch: BatchData, pred: PredictedData, margin: int | None = None
    ) -> LossOutput:
        """
        Evaluate against inputs
        """

        assert (
            batch.network_input.lr_tensor is not None
            and batch.network_input.lr_bands is not None
        )

        pred_bands, _ = match_bands(self.bands, pred.bands)
        assert len(pred_bands) == len(self.bands)
        target_bands, _ = match_bands(self.bands, batch.network_input.lr_bands)
        assert len(target_bands) == len(self.bands)

        if margin is None:
            actual_margin = pred.margin
        else:
            actual_margin = margin

        factor = (
            float(pred.prediction.shape[-1]) / batch.network_input.lr_tensor.shape[-1]
        )

        # Make sure actual margin can be divided by factor
        actual_margin = int(factor * np.ceil(actual_margin / factor))

        actual_lr_margin = int(actual_margin / factor)

        lr_loss = self.weight * self.loss(
            pred.prediction[
                :,
                pred_bands,
                actual_margin:-actual_margin,
                actual_margin:-actual_margin,
            ],
            batch.network_input.lr_tensor[
                :,
                target_bands,
                actual_lr_margin:-actual_lr_margin,
                actual_lr_margin:-actual_lr_margin,
            ],
        )

        return LossOutput(self.weight * lr_loss, self.bands)
