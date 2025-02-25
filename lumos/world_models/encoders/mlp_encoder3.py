from typing import List
import torch
from torch import nn
from lumos.utils.nn_utils import get_activation, flatten_batch, unflatten_batch


class MLPEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        mlp_layers: int,
        activation: str,
    ):
        super(MLPEncoder, self).__init__()
        self.state_dim = state_dim
        self.mlp_layers = mlp_layers
        self.activation = get_activation(activation)

        self.out_dim = output_dim

        # Encoder for both robot and scene observations
        modules = []
        in_channels = state_dim
        for h_dim in self.mlp_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    self.activation,
                )
            )
            in_channels = h_dim
        modules.append(nn.Linear(in_channels, self.out_dim))
        self.encoder = nn.Sequential(*modules)

    def forward(self, x_robot_scene):
        x_robot_scene, bd = flatten_batch(x_robot_scene, 1)
        y = self.encoder(x_robot_scene)
        y = unflatten_batch(y, bd)
        return y
