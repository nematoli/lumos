from typing import List

import torch
from torch import nn

from lumos.utils.nn_utils import flatten_batch, get_activation, unflatten_batch


class CnnEncoder(nn.Module):
    def __init__(
        self,
        cnn_depth: int,
        kernels: List[int],
        strides: List[int],
        paddings: List[int],
        activation: str,
        use_gripper_camera: bool,
    ):
        super(CnnEncoder, self).__init__()
        self.in_dim = 3
        self.cnn_depth = cnn_depth
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.activation = get_activation(activation)
        self.use_gripper_camera = use_gripper_camera
        self.out_dim = self.cnn_depth * (2 ** (len(self.kernels) + 1))

        self.encoder_static = self._make_encoder_layers()

        if self.use_gripper_camera:
            self.encoder_gripper = self._make_encoder_layers()
            self.fuse = nn.Sequential(
                nn.Linear(self.out_dim * 2, self.out_dim), nn.LayerNorm(self.out_dim, eps=1e-3), self.activation
            )

    def _make_encoder_layers(self):
        layers = []
        in_channels = self.in_dim

        for i in range(len(self.kernels)):
            out_channels = self.cnn_depth * (2**i)
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=self.paddings[i],
                )
            )
            layers.append(self.activation)
            in_channels = out_channels

        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def forward(self, x_s, x_g):
        x_s, bd = flatten_batch(x_s, 3)
        y_s = self.encoder_static(x_s)

        if x_g is not None:
            x_g, _ = flatten_batch(x_g, 3)
            y_g = self.encoder_gripper(x_g)
            y_combined = torch.cat([y_s, y_g], dim=1)
            y = self.fuse(y_combined)
        else:
            y = y_s

        y = unflatten_batch(y, bd)
        return y
