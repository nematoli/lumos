from typing import List
import torch
import torch.nn as nn
from lumos.utils.nn_utils import get_activation, NoNorm, flatten_batch, unflatten_batch


class MLPDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        robot_dim: int,
        scene_dim: int,
        activation: str,
        mlp_layers: list,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.robot_dim = robot_dim
        self.scene_dim = scene_dim
        self.mlp_layers = mlp_layers
        self.activation = get_activation(activation)
        self.mlp_layers = mlp_layers

        # Robot obs decoder
        modules = []
        in_channels = in_dim
        for h_dim in self.mlp_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    self.activation,
                )
            )
            in_channels = h_dim
        modules.append(nn.Linear(in_channels, self.robot_dim))
        self.decoder_robot = nn.Sequential(*modules)

        # Scene obs decoder
        modules = []
        in_channels = in_dim
        for h_dim in self.mlp_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    self.activation,
                )
            )
            in_channels = h_dim
        modules.append(nn.Linear(in_channels, self.scene_dim))
        self.decoder_scene = nn.Sequential(*modules)

    def forward(self, features):
        x, bd = flatten_batch(features)
        y_robot = self.decoder_robot(x)
        y_scene = self.decoder_scene(x)
        dcd_robot = unflatten_batch(y_robot, bd)
        dcd_scene = unflatten_batch(y_scene, bd)
        return dcd_robot, dcd_scene
