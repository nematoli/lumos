from typing import Tuple
import hydra
from omegaconf import DictConfig
import torch
import torch.distributions as D
import torch.nn as nn
from torch import Tensor


class RSSMCore(nn.Module):
    def __init__(self, cell: DictConfig):
        super().__init__()
        self.cell = hydra.utils.instantiate(cell)

    def forward(self, embeds: Tensor, actions: Tensor, resets: Tensor, in_state: Tuple[Tensor, Tensor]):
        reset_masks = ~resets
        T = embeds.shape[0]

        (h, z) = in_state
        posts, states_h, samples = [], [], []
        for i in range(T):
            post, (h, z) = self.cell.forward(actions[i], (h, z), reset_masks[i], embeds[i])
            posts.append(post)
            states_h.append(h)
            samples.append(z)

        posts = torch.stack(posts)
        states_h = torch.stack(states_h)
        samples = torch.stack(samples)
        priors = self.cell.batch_prior(states_h)
        features = self.to_feature(states_h, samples)

        return (priors, posts, features, (h.detach(), z.detach()))

    def init_state(self, batch_size):
        return self.cell.init_state(batch_size)

    def to_feature(self, h: Tensor, z: Tensor) -> Tensor:
        return torch.cat((h, z), -1)

    def feature_replace_z(self, features: Tensor, z: Tensor):
        h, _ = features.split([self.cell.deter_dim, z.shape[-1]], -1)
        return self.to_feature(h, z)

    def zdistr(self, pp: Tensor) -> D.Distribution:
        return self.cell.zdistr(pp)
