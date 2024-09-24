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

        # Decoder for both robot and scene observations
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
        modules.append(nn.Linear(in_channels, self.robot_dim + self.scene_dim))
        self.decoder = nn.Sequential(*modules)

    def forward(self, features):
        features, bd = flatten_batch(features)
        dcd_robot_scene = self.decoder(features)
        dcd_robot_scene = unflatten_batch(dcd_robot_scene, bd)
        return dcd_robot_scene
