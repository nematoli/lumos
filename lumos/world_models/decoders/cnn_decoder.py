from typing import List

import torch.nn as nn

from lumos.utils.nn_utils import flatten_batch, get_activation, NoNorm, unflatten_batch


class CnnDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        cnn_depth: int,
        kernels: List[int],
        strides: List[int],
        paddings: List[int],
        out_channels: List[int],
        layer_norm: bool,
        activation: str,
        mlp_layers: int,
        use_gripper_camera: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 3
        self.cnn_depth = cnn_depth
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.out_channels = out_channels
        self.norm = nn.LayerNorm if layer_norm else NoNorm
        self.activation = get_activation(activation)
        self.mlp_layers = mlp_layers
        self.use_gripper_camera = use_gripper_camera

        self.deepest_dim = self.cnn_depth * (2 ** (len(self.kernels) + 1))

        self.decoder_static = self._make_decoder_layers()

        if self.use_gripper_camera:
            self.decoder_gripper = self._make_decoder_layers()

    def _make_decoder_layers(self):
        out_channels_list = self.out_channels
        num_conv_layers = len(self.kernels)
        assert len(out_channels_list) == num_conv_layers, "Mismatch in kernel/stride/output channel list lengths"

        in_channels = self.deepest_dim

        if self.mlp_layers == 0:
            layers = [nn.Linear(self.in_dim, in_channels)]
        else:
            hidden_dim = in_channels
            layers = [nn.Linear(self.in_dim // 2, hidden_dim), self.norm(hidden_dim, eps=1e-3), self.activation]
            for _ in range(self.mlp_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), self.norm(hidden_dim, eps=1e-3), self.activation]

        decoder_layers = []
        for i in range(num_conv_layers):
            out_channels = out_channels_list[i]
            decoder_layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=self.paddings[i],
                )
            )
            if i < num_conv_layers - 1:
                decoder_layers.append(self.activation)
            in_channels = out_channels

        return nn.Sequential(*layers, nn.Unflatten(-1, (self.deepest_dim, 1, 1)), *decoder_layers)

    def forward(self, features):
        x, bd = flatten_batch(features)
        y_s = self.decoder_static(x)
        dcd_img_s = unflatten_batch(y_s, bd)

        dcd_img_g = None
        if self.use_gripper_camera:
            y_g = self.decoder_gripper(x)
            dcd_img_g = unflatten_batch(y_g, bd)

        return dcd_img_s, dcd_img_g
