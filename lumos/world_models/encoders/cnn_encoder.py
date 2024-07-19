from typing import List
import torch
from torch import nn
from lumos.utils.nn_utils import get_activation, flatten_batch, unflatten_batch


class CnnEncoder(nn.Module):
    def __init__(
        self,
        cnn_depth: int,
        kernels: List[int],
        stride: int,
        activation: str,
    ):
        super(CnnEncoder, self).__init__()
        self.in_dim = 3
        self.cnn_depth = cnn_depth
        self.kernels = kernels
        self.stride = stride
        self.activation = get_activation(activation)

        self.out_dim = cnn_depth * 32

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
            nn.Linear(self.out_dim * 2, self.out_dim), nn.LayerNorm(self.out_dim, eps=1e-3), self.activation
        )

    def forward(self, x_s, x_g):
        x_s, bd = flatten_batch(x_s, 3)
        x_g, bd = flatten_batch(x_g, 3)
        y_s = self.encoder_static(x_s)
        y_g = self.encoder_gripper(x_g)
        y = torch.cat([y_s, y_g], dim=1)
        y = self.fuse(y)
        y = unflatten_batch(y, bd)
        return y
