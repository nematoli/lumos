from typing import List
import torch
from torch import nn
from lumos.utils.nn_utils import get_activation, flatten_batch, unflatten_batch


class MLPEncoder(nn.Module):
    def __init__(
        self,
        robot_dim: int,
        scene_dim: int,
        output_dim: int,
        mlp_layers: int,
        activation: str,
    ):
        super(MLPEncoder, self).__init__()
        self.robot_dim = robot_dim
        self.scene_dim = scene_dim
        self.mlp_layers = mlp_layers
        self.activation = get_activation(activation)

        self.out_dim = output_dim

        # Robot obs encoder
        modules = []
        in_channels = robot_dim
        for h_dim in self.mlp_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    self.activation,
                )
            )
            in_channels = h_dim
        modules.append(nn.Linear(in_channels, self.out_dim))
        self.encoder_robot = nn.Sequential(*modules)

        # Scene obs encoder
        modules = []
        in_channels = scene_dim
        for h_dim in self.mlp_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    self.activation,
                )
            )
            in_channels = h_dim
        modules.append(nn.Linear(in_channels, self.out_dim))
        self.encoder_scene = nn.Sequential(*modules)

        self.fuse = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.out_dim), nn.LayerNorm(self.out_dim, eps=1e-3), self.activation
        )

    def forward(self, x_robot, x_scene):
        x_robot, bd = flatten_batch(x_robot, 1)
        x_scene, bd = flatten_batch(x_scene, 1)
        y_robot = self.encoder_robot(x_robot)
        y_scene = self.encoder_scene(x_scene)
        y = torch.cat([y_robot, y_scene], dim=1)
        y = self.fuse(y)
        y = unflatten_batch(y, bd)
        return y
