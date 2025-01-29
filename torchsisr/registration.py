# Code inspired from L1BSR repo : https://github.com/centreborelli/L1BSR/tree/master
"""
This module contains code for the registration UNet
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from torchsisr.unet import UNet


class UnetOpticalFlowEstimation(nn.Module):
    """
    Estimate optical flow with a Unet
    """

    def __init__(
        self,
        max_range: float = 10.0,
        depth: int = 4,
        start_filts: int = 64,
        min_skip_depth: int = 2,
    ):
        """
        Initializer
        """
        super().__init__()

        self.unet = UNet(
            in_channels=2,
            out_channels=32,
            depth=depth,
            min_skip_depth=min_skip_depth,
            start_filts=start_filts,
            up_mode="upsample",
            activation=torch.nn.functional.leaky_relu,
        )

        self.final_conv = nn.Conv2d(
            in_channels=32,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )

        self.max_range = max_range / math.sqrt(2)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        """
        # Apply instance norm to input data and stack
        in_data = torch.stack(
            (
                torch.nn.functional.instance_norm(left),
                torch.nn.functional.instance_norm(right),
            ),
            dim=1,
        )

        unet_out = torch.nn.functional.leaky_relu(self.unet(in_data))

        return torch.nn.functional.tanh(self.final_conv(unet_out)) * self.max_range


def simulate_warp(
    data: torch.Tensor,
    max_range: float = 10.0,
    max_width: float = 10.0,
) -> torch.Tensor:
    """
    Simulate warping
    """
    x = torch.arange(0, data.shape[-2], device=data.device)
    y = torch.arange(0, data.shape[-1], device=data.device)

    yy, xx = torch.meshgrid(x, y)

    phi = data.shape[-1] * torch.rand((data.shape[0], 2), device=data.device)
    omega = data.shape[-1] * (
        0.9 + 0.1 * torch.rand((data.shape[0], 2), device=data.device)
    )

    cos_weight = torch.rand((data.shape[0],), device=data.device)
    omega = 1 / omega
    dx = cos_weight[:, None, None] * torch.cos(
        (omega[:, 0, None, None] * (xx[None, ...] + phi[:, 0, None, None]))
        * (2 * math.pi)
    )
    dy = cos_weight[:, None, None] * torch.cos(
        (omega[:, 1, None, None] * (yy[None, ...] + phi[:, 1, None, None]))
        * (2 * math.pi)
    )

    a = 10 * 2 * (torch.rand((data.shape[0],), device="cuda") - 0.5)
    b = -(a * data.shape[-1] * torch.rand((data.shape[0],), device="cuda"))

    signed_dist = (
        yy[None, ...] - a[:, None, None] * xx[None, ...] - b[:, None, None]
    ) / torch.sqrt(1 + a[:, None, None] ** 2)

    width = 1.0 + (max_width - 1) * torch.rand((data.shape[0],), device="cuda")
    amp_x = 2 * (torch.rand((data.shape[0],), device="cuda") - 0.5)
    amp_y = 2 * (torch.rand((data.shape[0],), device="cuda") - 0.5)

    dx += amp_x[:, None, None] * (
        torch.nn.functional.tanh(signed_dist / width[:, None, None])
    )
    dy += amp_y[:, None, None] * (
        torch.nn.functional.tanh(signed_dist / width[:, None, None])
    )

    target_flow = torch.stack((dx, dy), dim=1)

    target_flow = max_range * target_flow

    return target_flow


def warp(x: torch.Tensor, flo: torch.Tensor) -> torch.Tensor:
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    if torch.sum(flo * flo) == 0:
        return x

    b, _, h, w = x.size()

    # mesh grid
    xx = torch.arange(0, w, device=x.device).view(1, -1).repeat(h, 1)
    yy = torch.arange(0, h, device=x.device).view(-1, 1).repeat(1, w)
    xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
    yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flo.to(x.device)

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(w - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(h - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(
        x, vgrid, align_corners=True, mode="bicubic", padding_mode="reflection"
    )
    return output


def compose_flows(flo1: torch.Tensor, flo2: torch.Tensor) -> torch.Tensor:
    """
    (code from https://github.com/centreborelli/L1BSR/blob/master/warpingOperator.py )
    compose flows flo1 and flo2
    flo: [B, 2, H, W] flow
    """

    b, _, h, w = flo1.size()

    # mesh grid
    xx = torch.arange(0, w, device=flo1.device).view(1, -1).repeat(h, 1)
    yy = torch.arange(0, h, device=flo1.device).view(-1, 1).repeat(1, w)
    xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
    yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    vgrid = grid + flo1

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(w - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(h - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    vgrid = F.grid_sample(flo2, vgrid, align_corners=True, mode="bilinear")

    return flo1 + flo2
