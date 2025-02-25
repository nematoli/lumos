from typing import List
import torch
import torch.nn as nn
from lumos.utils.nn_utils import get_activation, NoNorm, flatten_batch, unflatten_batch


class HybridDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        cnn_depth: int,
        kernels: List[int],
        stride: int,
        layer_norm: bool,
        activation: str,
        mlp_layers: int,
        state_mlp_layers: List[int],
        state_out_dim: int,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 3
        self.cnn_depth = cnn_depth
        self.kernels = kernels
        self.stride = stride
        self.norm = nn.LayerNorm if layer_norm else NoNorm
        self.activation = get_activation(activation)
        self.mlp_layers = mlp_layers
        self.state_mlp_layers = state_mlp_layers
        self.state_dim = state_out_dim

        if self.mlp_layers == 0:
            layers = [
                nn.Linear(self.in_dim, self.cnn_depth * 32),  # No activation here in DreamerV2
            ]
        else:
            hidden_dim = self.cnn_depth * 32
            layers = [nn.Linear(self.in_dim // 2, hidden_dim), self.norm(hidden_dim, eps=1e-3), self.activation]
            for _ in range(self.mlp_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), self.norm(hidden_dim, eps=1e-3), self.activation]

        self.decoder_static = nn.Sequential(
            *layers,
            nn.Unflatten(-1, (self.cnn_depth * 32, 1, 1)),
            nn.ConvTranspose2d(self.cnn_depth * 32, self.cnn_depth * 4, self.kernels[0], self.stride),
            self.activation,
            nn.ConvTranspose2d(self.cnn_depth * 4, self.cnn_depth * 2, self.kernels[1], self.stride),
            self.activation,
            nn.ConvTranspose2d(self.cnn_depth * 2, self.cnn_depth, self.kernels[2], self.stride),
            self.activation,
            nn.ConvTranspose2d(self.cnn_depth, self.out_dim, self.kernels[3], self.stride),
        )

        self.decoder_gripper = nn.Sequential(
            *layers,
            nn.Unflatten(-1, (self.cnn_depth * 32, 1, 1)),
            nn.ConvTranspose2d(self.cnn_depth * 32, self.cnn_depth * 4, self.kernels[0], self.stride),
            self.activation,
            nn.ConvTranspose2d(self.cnn_depth * 4, self.cnn_depth * 2, self.kernels[1], self.stride),
            self.activation,
            nn.ConvTranspose2d(self.cnn_depth * 2, self.cnn_depth, self.kernels[2], self.stride),
            self.activation,
            nn.ConvTranspose2d(self.cnn_depth, self.out_dim, self.kernels[3], self.stride),
        )

        # Decoder for both robot and scene observations
        modules = []
        in_channels = in_dim
        for h_dim in self.state_mlp_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    self.activation,
                )
            )
            in_channels = h_dim
        modules.append(nn.Linear(in_channels, self.state_dim))
        self.decoder_state = nn.Sequential(*modules)

    def forward(self, features):
        x, bd = flatten_batch(features)
        y_s = self.decoder_static(x)
        y_g = self.decoder_gripper(x)
        y_state = self.decoder_state(x)
        dcd_img_s = unflatten_batch(y_s, bd)
        dcd_img_g = unflatten_batch(y_g, bd)
        dcd_state = unflatten_batch(y_state, bd)
        return dcd_img_s, dcd_img_g, dcd_state
