# Adapted from /raid/scratch/mevans/o2vae/models/decoders/cnn_decoder.py
# Upstream commit: 4c70d589954a86d114aba2d27ff7111573ac669c4
# License/attribution: retain upstream O2-VAE attribution and repository license.
# Changes: derive a 19-pixel base map and exact power-of-two output size from
# image_size; use four 2x stages for 304x304 instead of a 1x1 power-of-two path.

import torch
import torch.nn as nn


class CnnDecoder(nn.Module):
    def __init__(
        self,
        zdim,
        cnn_dims=(64, 32, 16, 8),
        out_channels=1,
        base_size=19,
        image_size=304,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        last_kernel_size=None,
        **kwargs,
    ):
        """
        Args:
            last_kernel_size (int): allows to control the very last conv layer before output
        """
        super().__init__()
        if len(cnn_dims) < 1:
            raise ValueError("cnn_dims must contain at least one channel width")
        self.base_size = int(base_size)
        self.image_size = int(image_size)
        self.n_stages = len(cnn_dims)
        expected_size = self.base_size * (2 ** self.n_stages)
        self.pad_output = self.image_size == expected_size + 1
        if expected_size != self.image_size and not self.pad_output:
            raise ValueError(
                f"image_size={self.image_size} must equal "
                f"base_size * 2**n_stages ({self.base_size} * 2**{self.n_stages}={expected_size}) "
                "or that size plus one for odd padding"
            )
        activation = nn.ELU()
        self.fc = nn.Sequential(
            nn.Linear(zdim, int(cnn_dims[0]) * self.base_size * self.base_size),
            nn.ELU(),
        )

        layers = []
        in_dim = int(cnn_dims[0])

        for dim in [*cnn_dims[1:], out_channels]:
            dim = int(dim)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            layers.append(nn.BatchNorm2d(dim))
            layers.append(activation)
            in_dim = dim

        self.dec_conv = nn.Sequential(*layers)

    def forward(self, x):
        bs = x.size(0)
        x = self.fc(x)
        x = x.view(bs, self.fc[0].out_features // (self.base_size ** 2), self.base_size, self.base_size)
        x = self.dec_conv(x)

        if self.pad_output:
            x = nn.functional.pad(x, (0, 1, 0, 1))
        return x
