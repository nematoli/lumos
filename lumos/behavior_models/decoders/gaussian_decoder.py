import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from omegaconf import ListConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import lumos
from lumos.behavior_models.decoders.action_decoder import ActionDecoder
from lumos.behavior_models.actor_critic.mlp import MLP

logger = logging.getLogger(__name__)


class GaussianDecoder(ActionDecoder):
    def __init__(
        self,
        perceptual_features: int,
        latent_goal_features: int,
        plan_features: int,
        hidden_size: int,
        layers: int,
        layer_norm: bool,
        activation: str,
        out_features: int,
        log_std_min: float,
        log_std_max: float,
    ):
        super(GaussianDecoder, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.plan_features = plan_features

        in_features = plan_features + perceptual_features + latent_goal_features
        self.out_features = out_features  # for discrete gripper act

        self.fc = MLP(in_features, None, hidden_size, layers, layer_norm, activation)

        self.actor_mean = MLP(in_features, self.out_features, hidden_size, 0, layer_norm, activation)
        self.actor_std = MLP(in_features, self.out_features, hidden_size, 0, layer_norm, activation)

    # Sampling from logistic distribution
    def _sample(self, log_stds: Tensor, means: Tensor) -> Tuple[Tensor, Tensor]:

        action_stds = torch.exp(log_stds)
        a_dist = torch.distributions.Normal(means, action_stds)
        z = a_dist.rsample()
        actions = torch.tanh(z)
        entropy = a_dist.entropy()

        return actions, entropy

    def forward(  # type: ignore
        self,
        latent_current: Tensor,
        latent_goal: Tensor,
        latent_plan: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        if latent_plan is not None:
            x = torch.cat([latent_current, latent_goal, latent_plan], dim=-1)  # b, s, (plan + visuo-propio + goal)
        else:
            x = torch.cat([latent_current, latent_goal], dim=-1)

        x = self.fc(x)

        means = self.actor_mean(x)
        log_stds = self.actor_std(x)

        log_stds = torch.tanh(log_stds)
        log_stds = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_stds + 1)

        return log_stds, means

    def get_action(
        self,
        latent_current: Tensor,
        latent_goal: Tensor,
        latent_plan: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        log_stds, means = self(latent_current, latent_goal, latent_plan)
        actions, entropy = self._sample(log_stds, means)
        return actions, entropy
