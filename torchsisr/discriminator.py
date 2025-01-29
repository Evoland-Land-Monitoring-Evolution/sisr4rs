#!/usr/bin/env python
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
"""
Contains the discriminator from ESRGAN
"""

import torch

from torchsisr.dataset import high_pass_filtering


class ConvBlock(torch.nn.Module):
    """
    Convolution block for Discriminator
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        batch_norm: bool = True,
        kernel_size=(3, 3),
    ):
        """
        Initializer
        """
        super().__init__()

        self.net = torch.nn.Sequential()

        self.net.append(
            torch.nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                bias=False,
                padding="valid",
            )
        )
        if batch_norm:
            self.net.append(torch.nn.BatchNorm2d(out_features))

        self.net.append(torch.nn.LeakyReLU(negative_slope=0.2))

    def forward(self, in_data: torch.Tensor):
        """
        Forward method
        """
        return self.net(in_data)


class ConvEncoder(torch.nn.Module):
    """
    Encoder for discriminator
    """

    def __init__(
        self,
        in_features: int = 4,
        nb_additional_latent_layers: int = 2,
        latent_features: int = 64,
        batch_norm: bool = True,
        kernel_size: tuple[int, int] = (3, 3),
    ):
        """ """
        super().__init__()
        self.net = torch.nn.Sequential()

        # At least one block
        self.net.append(
            ConvBlock(
                in_features,
                latent_features,
                batch_norm=batch_norm,
                kernel_size=kernel_size,
            )
        )

        # Add subsequent blocks
        for _ in range(nb_additional_latent_layers):
            self.net.append(
                ConvBlock(
                    latent_features,
                    latent_features,
                    batch_norm=batch_norm,
                    kernel_size=kernel_size,
                )
            )

    def forward(self, in_data: torch.Tensor):
        """
        Forward method
        """
        return self.net(in_data)


class LinearHead(torch.nn.Module):
    """
    Linear head for classification
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int = 256,
        nb_additional_latent_layers: int = 1,
        out_features: int = 1,
    ):
        """ """
        super().__init__()

        self.net = torch.nn.Sequential()

        # Add first layer
        self.net.append(torch.nn.Linear(in_features, latent_features))
        self.net.append(torch.nn.LeakyReLU())

        # Add subsequent layers
        for _ in range(nb_additional_latent_layers):
            self.net.append(torch.nn.Linear(latent_features, latent_features))
            self.net.append(torch.nn.LeakyReLU())

        # Last layer
        self.net.append(torch.nn.Linear(latent_features, out_features))

    def forward(self, in_data: torch.Tensor):
        """
        Forward method
        """
        return self.net(in_data)


class Discriminator(torch.nn.Module):
    """
    Discriminator class
    """

    def __init__(
        self,
        in_features: int = 4,
        encoder_latent_features: int = 32,
        nb_additional_latent_layers: int = 0,
        head_latent_features: int = 512,
        head_nb_additional_latent_layers: int = 0,
        pooling_size: int = 2,
        high_pass_filtering_mtf: float | None = None,
    ):
        """ """
        super().__init__()
        self.high_pass_filtering_mtf = high_pass_filtering_mtf

        self.encoder = ConvEncoder(
            in_features=in_features,
            latent_features=encoder_latent_features,
            nb_additional_latent_layers=nb_additional_latent_layers,
        )

        self.pooling = torch.nn.AdaptiveAvgPool2d(
            output_size=(pooling_size, pooling_size)
        )

        self.head = LinearHead(
            in_features=pooling_size * pooling_size * encoder_latent_features,
            latent_features=head_latent_features,  # 7x7x64
            nb_additional_latent_layers=head_nb_additional_latent_layers,
            out_features=1,
        )

    def forward(self, in_data: torch.Tensor):
        """
        Forward method
        """
        if self.high_pass_filtering_mtf is not None:
            in_data = high_pass_filtering(
                in_data, mtf=self.high_pass_filtering_mtf, scale_factor=2.0
            )

        return self.head(self.pooling(self.encoder(in_data)).flatten(1, -1))


# Taken from : https://github.com/xinntao/Real-ESRGAN/blob/master/
# realesrgan/archs/discriminator_arch.py
class UNetDiscriminatorSN(torch.nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind
    Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.

        skip_connection (bool): Whether to use skip connections between
        U-Net. Default: True.

    """

    def __init__(
        self,
        num_in_ch,
        num_feat=64,
        skip_connection=True,
        high_pass_filtering_mtf: float | None = None,
    ):
        super().__init__()
        self.skip_connection = skip_connection
        self.high_pass_filtering_mtf = high_pass_filtering_mtf
        norm = torch.nn.utils.spectral_norm
        # the first convolution
        self.conv0 = torch.nn.Conv2d(
            num_in_ch, num_feat, kernel_size=3, stride=1, padding=1
        )
        # downsample
        self.conv1 = norm(torch.nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(
            torch.nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False)
        )
        self.conv3 = norm(
            torch.nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False)
        )
        # upsample
        self.conv4 = norm(
            torch.nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False)
        )
        self.conv5 = norm(
            torch.nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False)
        )
        self.conv6 = norm(torch.nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = torch.nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        """
        Forward method
        """
        # HPF if required
        if self.high_pass_filtering_mtf is not None:
            x = high_pass_filtering(
                x, mtf=self.high_pass_filtering_mtf, scale_factor=2.0
            )

        # downsample
        x0 = torch.nn.functional.leaky_relu(
            self.conv0(x), negative_slope=0.2, inplace=True
        )
        x1 = torch.nn.functional.leaky_relu(
            self.conv1(x0), negative_slope=0.2, inplace=True
        )
        x2 = torch.nn.functional.leaky_relu(
            self.conv2(x1), negative_slope=0.2, inplace=True
        )
        x3 = torch.nn.functional.leaky_relu(
            self.conv3(x2), negative_slope=0.2, inplace=True
        )

        # upsample
        x3 = torch.nn.functional.interpolate(
            x3, size=x2.shape[-2:], mode="bilinear", align_corners=False
        )
        x4 = torch.nn.functional.leaky_relu(
            self.conv4(x3), negative_slope=0.2, inplace=True
        )

        if self.skip_connection:
            x4 = x4 + x2
        x4 = torch.nn.functional.interpolate(
            x4, size=x1.shape[-2:], mode="bilinear", align_corners=False
        )
        x5 = torch.nn.functional.leaky_relu(
            self.conv5(x4), negative_slope=0.2, inplace=True
        )
        if self.skip_connection:
            x5 = x5 + x1
        x5 = torch.nn.functional.interpolate(
            x5, size=x0.shape[-2:], mode="bilinear", align_corners=False
        )
        x6 = torch.nn.functional.leaky_relu(
            self.conv6(x5), negative_slope=0.2, inplace=True
        )

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = torch.nn.functional.leaky_relu(
            self.conv7(x6), negative_slope=0.2, inplace=True
        )
        out = torch.nn.functional.leaky_relu(
            self.conv8(out), negative_slope=0.2, inplace=True
        )
        out = self.conv9(out)

        return out
