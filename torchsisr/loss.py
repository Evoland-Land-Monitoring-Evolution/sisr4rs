#!/usr/bin/env python
"""
Helper classes and custom losses
"""

from typing import Literal

import numpy as np
import torch
import torchmetrics
from piq import BRISQUELoss, brisque, dss  # type: ignore
from torch import nn

from torchsisr.dataset import generic_downscale, high_pass_filtering

DEFAULT_SMOOTH_L1_LOSS = torch.nn.SmoothL1Loss()
DEFAULT_SQUARED_ERROR_LOSS = torchmetrics.MeanSquaredError(squared=False)


def set_reduction(loss_wrapper):
    """
    Function that is used to set the reduction of loss function to
    'none' (when getting indices with shift wrapper) or back to its
    default value that can be 'mean' or 'elementwise_mean' when the
    shifted indices are known, but the loss function class attributes
    are still set to the "return_indices" task
    """
    if hasattr(loss_wrapper.loss, "reduction"):
        if loss_wrapper.reduction == "none":
            loss_wrapper.loss.reduction = loss_wrapper.reduction
        elif (
            loss_wrapper.reduction != "none"
            and "mean" not in loss_wrapper.loss.reduction
        ):
            default = loss_wrapper.loss.__class__().reduction
            loss_wrapper.reduction = default
            loss_wrapper.loss.reduction = default


class ShiftWrapper(nn.Module):
    """
    Wrapper to compute shifted loss/metrics
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        shift: int = 4,
        optimize_min_max: Literal["max", "min"] = "min",
        return_indices: bool = True,
    ):
        """
        Constructor
        :shift: Shift value in pixels
        :loss_fn: loss/metric function used
        :optimize_min_max: if we want to minimize or maximize metric value
        :return_indices: if we compute the best shift for each patch and return it
        """
        super().__init__()

        self.shift = shift
        self.loss = loss_fn
        self.optimize_min_max = optimize_min_max
        self.return_indices = return_indices

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        precomputed_indices: torch.Tensor | None = None,
    ):
        """
        Forward method
        """
        size_patch = pred.shape[-2], pred.shape[-1]
        cropped_target = target[
            :,
            :,
            self.shift : size_patch[0] - self.shift,
            self.shift : size_patch[1] - self.shift,
        ]

        # We search the best shifted loss value
        if precomputed_indices is None:
            # All shifted losses
            all_losses_list: list[torch.Tensor] = []
            self.loss.reduction = "none"  # type: ignore

            # We compute loss/metric for each shifted pair
            shift_counter = 0
            for i in range((2 * self.shift) + 1):
                for j in range((2 * self.shift) + 1):
                    cropped_predictions = pred[
                        :,
                        :,
                        i : i + (size_patch[0] - (2 * self.shift)),
                        j : j + (size_patch[1] - (2 * self.shift)),
                    ]
                    shifted_loss = self.loss(cropped_predictions, cropped_target)
                    all_losses_list.append(shifted_loss)
                    shift_counter += 1
            all_losses = torch.stack(all_losses_list)
            # we reduce loss value per band [nb_shifted_losses, bands,
            # batch] or [nb_shifted_losses, batch] if global loss
            all_losses = (
                all_losses.mean((-1, -2)) if all_losses.dim() > 3 else all_losses
            )

            # We compute the best shifted indices if needed
            # all_losses.dim() == 4 - metric is computed per band and
            # we first need to compute the mean per patch before
            # getting the best shift index else (all_losses.dim() ==
            # 1): metric is computed on global level, nothing else
            # needed all_losses[indices].mean(-1) means that we reduce
            # per batch
            # if metric should be maximized or minimized
            if self.optimize_min_max == "min":
                indices = (
                    all_losses.mean(1).min(0)[1]
                    if all_losses.dim() == 3
                    else all_losses.min(0)[1]
                )
            else:
                indices = (
                    all_losses.mean(1).max(0)[1]
                    if all_losses.dim() == 3
                    else all_losses.max(0)[1]
                )
            indices_exp = (
                indices.unsqueeze(-1)
                .expand(all_losses.shape[2], all_losses.shape[1])
                .unsqueeze(1)
            )
            best_loss = (
                torch.gather(all_losses.permute(2, 0, 1), dim=1, index=indices_exp)
                .squeeze(1)
                .mean(0)
            )
            return (best_loss, indices) if self.return_indices else best_loss

        # We extract the shifted images
        shift_counter = 0
        all_cropped_predictions = torch.zeros_like(cropped_target)
        for i in range((2 * self.shift) + 1):
            for j in range((2 * self.shift) + 1):
                if precomputed_indices.max() < shift_counter:
                    break
                cropped_predictions = pred[
                    precomputed_indices == shift_counter,
                    :,
                    i : i + (size_patch[0] - (2 * self.shift)),
                    j : j + (size_patch[1] - (2 * self.shift)),
                ]
                all_cropped_predictions[precomputed_indices == shift_counter] = (
                    cropped_predictions
                )
                shift_counter += 1
        shifted_loss = self.loss(all_cropped_predictions, cropped_target)
        return shifted_loss


class PerBandWrapper(nn.Module):
    """

    Wrapper to compute metrics per band, assuming batches of [n,c,w,h]
    with c indexing bands

    """

    def __init__(
        self,
        loss: torch.nn.Module = DEFAULT_SMOOTH_L1_LOSS,
        dimension_4_needed: bool = False,
        clip: tuple[float, float] | None = None,
        reduction="mean",
    ):
        super().__init__()

        self.loss = loss
        self.reduction = reduction
        self.dimension_4_needed = dimension_4_needed
        self.clip = clip

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Forward call for Module
        """
        values = []

        set_reduction(self)

        # Apply loss to each band
        for band in range(pred.shape[1]):
            current_pred = pred[:, band, ...]
            current_target = target[:, band, ...]
            if self.dimension_4_needed:
                current_pred = current_pred.unsqueeze(1)
                current_target = current_target.unsqueeze(1)
            if self.clip is not None:
                current_pred = torch.clip(current_pred, self.clip[0], self.clip[1])
                current_target = torch.clip(current_target, self.clip[0], self.clip[1])
            values.append(self.loss(current_pred, current_target))
        return torch.stack(values)


class InverseLossHelper(nn.Module):
    """
    A simple class to inverse a loss function that should be maximized
    """

    def __init__(self, loss: nn.Module):
        """
        Class constructor
        """
        super().__init__()
        self.loss = loss

    def forward(self, ref: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        The forward method
        """
        return 1.0 - self.loss(ref, pred)


class PeakSignalNoiseRatio(nn.Module):
    """
    The PeakSignalNoiseRatio metric
    """

    def __init__(self, reduction: str = "mean", data_range: float = 1.0):
        """
        Initializer
        """
        super().__init__()
        self.data_range = torch.tensor(data_range)
        self.base = 10
        self.loss = torch.nn.MSELoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, ref: torch.Tensor):
        """
        Forward method
        """
        set_reduction(self)
        mse = self.loss(pred, ref)
        if self.reduction == "none":
            mse = mse.mean((-1, -2))
        psnr_base_e = 2 * torch.log(self.data_range) - torch.log(mse)
        psnr = psnr_base_e * (self.base / torch.log(torch.tensor(self.base)))
        return psnr


class LRFidelity(nn.Module):
    """
    Compute LRFidelity
    """

    def __init__(
        self,
        factor: float,
        mtf: float | None,
        downsampling_mode: str = "bicubic",
        loss: torch.nn.Module = DEFAULT_SQUARED_ERROR_LOSS,
        per_band: bool = True,
    ):
        """
        Constructor
        :param loss: Loss instance to use to compute LR fidelity
        :per_band: If we compute the metric per band or on global level
        """
        super().__init__()
        self.loss = loss
        self.factor = factor
        self.mtf = mtf
        self.downsampling_mode = downsampling_mode
        self.per_band = per_band

    def forward(self, pred: torch.Tensor, ref_lr: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        """
        if self.mtf is not None:
            pred_lr = generic_downscale(
                pred,
                factor=self.factor,
                mtf=self.mtf,
                padding="valid",
                mode=self.downsampling_mode,
            )
        else:
            pred_lr = torch.nn.functional.interpolate(
                pred, scale_factor=1 / self.factor, mode="area"
            )

        if self.per_band:
            values = [
                self.loss(
                    pred_lr[:, band, ...].contiguous(),
                    ref_lr[:, band, ...].contiguous(),
                )
                for band in range(pred_lr.shape[1])
            ]
            return torch.stack(values)
        return self.loss(pred_lr.contiguous(), ref_lr.contiguous())


class RMSELoss(nn.Module):
    """
    Compute RMSE loss
    """

    def __init__(self, reduction: str = "mean"):
        """
        Constructor
        :param loss: Loss instance to use to compute RMSE loss
        """
        super().__init__()
        self.reduction = reduction
        self.loss: torch.nn.Module = torch.nn.MSELoss(reduction=self.reduction)

    def forward(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Module forward method
        """
        set_reduction(self)
        if self.reduction == "none":
            return torch.sqrt(self.loss(pred, ref).mean((-1, -2)))
        return torch.sqrt(self.loss(pred, ref))


DEFAULT_RMSE_LOSS = RMSELoss()


class HRFidelity(nn.Module):
    """
    Compute HRFidelity
    """

    def __init__(
        self,
        factor: float,
        mtf: float,
        loss: torch.nn.Module = DEFAULT_RMSE_LOSS,
        reduction: str = "mean",
    ):
        """
        Constructor

        :param loss: Loss instance to use to compute HR fidelity
        """
        super().__init__()
        self.loss = loss
        self.mtf = mtf
        self.factor = factor

        self.reduction = reduction

    def forward(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Module forward method
        """

        set_reduction(self)

        pred_hf = high_pass_filtering(pred, self.mtf, self.factor)
        ref_hf = high_pass_filtering(ref, self.mtf, self.factor)

        values = [
            self.loss(pred_hf[:, band, ...], ref_hf[:, band, ...])
            for band in range(pred_hf.shape[1])
        ]

        return torch.stack(values)


class GradientStrataWrapper(nn.Module):
    """
    Compute gradient strata loss
    """

    def __init__(
        self,
        grad_mag_min: float,
        grad_mag_max: float,
        max_channel: int = 4,
        loss: torch.nn.Module = DEFAULT_RMSE_LOSS,
    ):
        """
        Constructor

        :param loss: Loss instance to use to compute HR fidelity
        :param grad_mag_min: Minimum of gradient magnitude to be used in loss
        :param grad_mag_max: Maximum of gradient magnitude to be used in loss
        """
        super().__init__()
        self.loss = loss
        self.max_channel = max_channel
        self.grad_mag_min = grad_mag_min
        self.grad_mag_max = grad_mag_max

    def forward(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Implement forward method
        """
        # Compute gradient magnitude
        grad_x, grad_y = torch.gradient(ref[:, : self.max_channel, ...], dim=(-1, -2))

        # Accumulate accross all bands
        grad_x = grad_x.mean(dim=1)
        grad_y = grad_y.mean(dim=1)

        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        grad_strata_mask = torch.logical_and(
            grad_magnitude >= self.grad_mag_min, grad_magnitude < self.grad_mag_max
        )

        assert grad_strata_mask.sum() > 0

        values = [
            self.loss(
                pred[:, band, ...][grad_strata_mask],
                ref[:, band, ...][grad_strata_mask],
            )
            for band in range(pred.shape[1])
        ]

        return torch.stack(values).to(dtype=torch.float32)


class FFTHFPowerVariation(nn.Module):
    """
    Compute FFTPowerVariation
    """

    def __init__(self, scale_factor: float = 1.0, support: float = 0.25):
        """
        Constructor

        :param loss: Loss instance to use to compute HR fidelity
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.support = support

    def fft(self, data: torch.Tensor):
        """
        Custom fft
        """
        out_fft = torch.cat(
            [
                torch.abs(
                    # pylint: disable=not-callable
                    torch.fft.fftshift(
                        # pylint: disable=not-callable
                        torch.fft.fft2(
                            data[None, i, ...].to(dtype=torch.float32), norm="backward"
                        )
                    )
                )
                for i in range(data.shape[0])
            ],
            dim=0,
        )

        norm = out_fft.shape[-1] * out_fft.shape[-2]

        return out_fft / norm

    def forward(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Module forward method
        """
        if self.scale_factor > 1:
            ref = torch.nn.functional.interpolate(
                ref, scale_factor=self.scale_factor, mode="bicubic", align_corners=False
            )

        pred_fft = self.fft(pred)
        ref_fft = self.fft(ref)

        hf_support_mult = int(np.ceil(2 / self.support))

        hf_lower_bounds_x = (ref_fft.shape[-1] // 2) - (
            ref_fft.shape[-1] // hf_support_mult
        )
        hf_lower_bounds_y = (ref_fft.shape[-2] // 2) - (
            ref_fft.shape[-2] // hf_support_mult
        )
        hf_upper_bounds_x = (ref_fft.shape[-1] // 2) + (
            ref_fft.shape[-1] // hf_support_mult
        )
        hf_upper_bounds_y = (ref_fft.shape[-2] // 2) + (
            ref_fft.shape[-2] // hf_support_mult
        )
        ref_hf_power = ref_fft[:, :, :hf_lower_bounds_x, :].sum(axis=(0, -1, -2))
        ref_hf_power += ref_fft[:, :, hf_upper_bounds_x:, :].sum(axis=(0, -1, -2))
        ref_hf_power += ref_fft[
            :, :, hf_lower_bounds_x:hf_upper_bounds_x, :hf_lower_bounds_y
        ].sum(axis=(0, -1, -2))
        ref_hf_power += ref_fft[
            :, :, hf_lower_bounds_x:hf_upper_bounds_x, hf_upper_bounds_y:
        ].sum(axis=(0, -1, -2))

        ref_hf_power /= ref.shape[0]

        pred_hf_power = pred_fft[:, :, :hf_lower_bounds_x, :].sum(axis=(0, -1, -2))
        pred_hf_power += pred_fft[:, :, hf_upper_bounds_x:, :].sum(axis=(0, -1, -2))
        pred_hf_power += pred_fft[
            :, :, hf_lower_bounds_x:hf_upper_bounds_x, :hf_lower_bounds_y
        ].sum(axis=(0, -1, -2))
        pred_hf_power += pred_fft[
            :, :, hf_lower_bounds_x:hf_upper_bounds_x, hf_upper_bounds_y:
        ].sum(axis=(0, -1, -2))

        pred_hf_power /= pred.shape[0]
        eps = 1e-6
        percent_variation = 100 * (pred_hf_power - ref_hf_power) / (ref_hf_power + eps)

        return percent_variation


class PerBandBRISQUE(nn.Module):
    """
    Compute BRISQUE on predicted data
    """

    def __init__(self, shift: float = 0.0, scale: float = 1.0, data_range: float = 1.0):
        """
        Constructor
        """
        super().__init__()
        self.shift = shift
        self.scale = scale
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, _ref: torch.Tensor) -> torch.Tensor:
        """
        Module forward method
        """
        # Move tensor to cpu

        brisque_list = [
            brisque(
                torch.clip(
                    self.scale * (self.shift + pred[:, j : j + 1, ...]),
                    0.0,
                    self.data_range,
                ),
                data_range=self.data_range,
            )
            for j in range(pred.shape[1])
        ]

        return torch.stack(brisque_list)


class PerBandDSS(nn.Module):
    """
    Compute DSS on predicted data
    """

    def __init__(
        self,
        loss_mode: bool = False,
        shift: float = 0,
        scale: float = 1.0,
        data_range: float = 1.0,
    ):
        """
        Constructor
        """
        self.loss_mode = loss_mode
        self.shift = shift
        self.scale = scale
        self.data_range = data_range

        super().__init__()

    def forward(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Module forward method
        """
        # Move tensor to cpu
        dss_list = [
            dss(
                (pred[:, j : j + 1, ...] + self.shift) * self.scale,
                (ref[:, j : j + 1, ...] + self.shift) * self.scale,
                data_range=self.data_range,
            )
            for j in range(pred.shape[1])
        ]
        if self.loss_mode:
            dss_list = [torch.clip(1 - d, 0.0, 1.0) for d in dss_list]

        return torch.stack(dss_list)


class PerBandBRISQUEVariation(nn.Module):
    """
    Compute BRISQUEVariation on predicted data
    """

    def __init__(self, scale_factor: float = 1.0):
        """
        Constructor
        """
        super().__init__()
        self.brisque = BRISQUELoss(data_range=1.0, reduction="none")
        self.scale_factor = scale_factor

    def forward(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Module forward method
        """
        if self.scale_factor > 1:
            ref = torch.nn.functional.interpolate(
                ref, scale_factor=self.scale_factor, mode="bicubic", align_corners=False
            )

        brisque_var: list[torch.Tensor] = []

        for j in range(pred.shape[1]):
            brisque_pred = self.brisque(torch.clip(pred[:, j : j + 1, ...], 0.0, 1.0))
            brisque_ref = self.brisque(torch.clip(ref[:, j : j + 1, ...], 0.0, 1.0))
            brisque_var.append(torch.mean(brisque_pred - brisque_ref, dim=0))

        return torch.stack(brisque_var)


class PerBandTVVariation(nn.Module):
    """
    Compute BRISQUEVariation on predicted data
    """

    def __init__(self, scale_factor: float = 1.0, absolute: bool = True):
        """
        Constructor
        """
        super().__init__()
        self.tv = torchmetrics.image.TotalVariation()
        self.scale_factor = scale_factor
        self.absolute = absolute

    def forward(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Module forward method
        """
        if self.scale_factor > 1:
            ref = torch.nn.functional.interpolate(
                ref, scale_factor=self.scale_factor, mode="bicubic", align_corners=False
            )

        tv_var_sum = torch.zeros((pred.shape[1],), device=pred.device, dtype=pred.dtype)

        for j in range(pred.shape[1]):
            tv_pred = self.tv(pred[:, j : j + 1, ...])

            tv_ref = self.tv(ref[:, j : j + 1, ...])
            if self.absolute:
                tv_var_sum[j] += torch.abs(tv_pred - tv_ref) / tv_ref
            else:
                tv_var_sum[j] += 100 * (tv_pred - tv_ref) / tv_ref
        return tv_var_sum / pred.shape[0]
