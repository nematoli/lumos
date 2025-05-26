from typing import List

import torch
from torch import nn

from lumos.utils.nn_utils import flatten_batch, get_activation, unflatten_batch


class CnnMLPEncoder(nn.Module):
    def __init__(
        self,
        cnn_depth: int,
        kernels: List[int],
        stride: int,
        activation: str,
        use_gripper_camera: bool,
        robot_dim: int,
        scene_dim: int,
        state_out_dim: int,
        state_mlp_layers: int,
    ):
        super().__init__()
        self.in_dim = 3
        self.cnn_depth = cnn_depth
        self.kernels = kernels
        self.stride = stride
        self.activation = get_activation(activation)
        self.use_gripper_camera = use_gripper_camera
        self.cnn_out_dim = cnn_depth * 32

        self.robot_dim = robot_dim
        self.scene_dim = scene_dim
        self.state_mlp_layers = state_mlp_layers
        self.state_out_dim = state_out_dim

        self.encoder_static = nn.Sequential(
            nn.Conv2d(self.in_dim, self.cnn_depth, self.kernels[0], self.stride),
            self.activation,
            nn.Conv2d(self.cnn_depth, self.cnn_depth * 2, self.kernels[1], self.stride),
            self.activation,
            nn.Conv2d(self.cnn_depth * 2, self.cnn_depth * 4, self.kernels[2], self.stride),
            self.activation,
            nn.Conv2d(self.cnn_depth * 4, self.cnn_depth * 8, self.kernels[3], self.stride),
            self.activation,
            nn.Flatten(),
        )
        if self.use_gripper_camera:
            self.encoder_gripper = nn.Sequential(
                nn.Conv2d(self.in_dim, self.cnn_depth, self.kernels[0], self.stride),
                self.activation,
                nn.Conv2d(self.cnn_depth, self.cnn_depth * 2, self.kernels[1], self.stride),
                self.activation,
                nn.Conv2d(self.cnn_depth * 2, self.cnn_depth * 4, self.kernels[2], self.stride),
                self.activation,
                nn.Conv2d(self.cnn_depth * 4, self.cnn_depth * 8, self.kernels[3], self.stride),
                self.activation,
                nn.Flatten(),
            )

            self.fuse = nn.Sequential(
                nn.Linear(self.cnn_out_dim * 2, self.cnn_out_dim),
                nn.LayerNorm(self.cnn_out_dim, eps=1e-3),
                self.activation,
            )

        # Encoder for robot + scene observations
        modules = []
        in_channels = robot_dim + scene_dim
        for h_dim in self.state_mlp_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    self.activation,
                )
            )
            in_channels = h_dim
        modules.append(nn.Linear(in_channels, self.state_out_dim))
        self.encoder_state = nn.Sequential(*modules)

    def forward(self, x_s, x_g, x_state):
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

        x_state, bd = flatten_batch(x_state, 1)
        y_state = self.encoder_state(x_state)
        y_state = unflatten_batch(y_state, bd)

        y_combined = torch.cat([y, y_state], dim=-1)
        return y_combined
