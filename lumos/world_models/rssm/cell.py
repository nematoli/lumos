import hydra
from omegaconf import DictConfig
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lumos.utils.nn_utils import NoNorm
from typing import Optional, Tuple


class RSSMCell(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deter_dim: int,
        stoch_dim: int,
        stoch_rank: int,
        hidden_dim: int,
        gru: DictConfig,
        layer_norm: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_rank = stoch_rank
        self.hidden_dim = hidden_dim
        self.gru = hydra.utils.instantiate(gru)

        self.norm = nn.LayerNorm if layer_norm else NoNorm

        self.init_h = nn.Parameter(torch.zeros((self.deter_dim)))
        self.init_z = nn.Parameter(torch.zeros((self.stoch_dim * self.stoch_rank)))

        self.z_mlp = nn.Linear(self.stoch_dim * (self.stoch_rank or 1), self.hidden_dim)
        self.a_mlp = nn.Linear(self.action_dim, self.hidden_dim, bias=False)
        self.in_norm = self.norm(self.hidden_dim, eps=1e-3)

        self.prior_mlp_h = nn.Linear(self.deter_dim, self.hidden_dim)
        self.prior_norm = self.norm(self.hidden_dim, eps=1e-3)
        self.prior_mlp = nn.Linear(self.hidden_dim, self.stoch_dim * (self.stoch_rank or 2))

        self.post_mlp_h = nn.Linear(self.deter_dim, self.hidden_dim)
        self.post_mlp_e = nn.Linear(self.embed_dim, self.hidden_dim, bias=False)
        self.post_norm = self.norm(self.hidden_dim, eps=1e-3)
        self.post_mlp = nn.Linear(self.hidden_dim, self.stoch_dim * (self.stoch_rank or 2))

    def init_state(self, batch_size):
        return (torch.tile(self.init_h, (batch_size, 1)), torch.tile(self.init_z, (batch_size, 1)))

    def forward(
        self,
        action: Tensor,
        in_state: Tuple[Tensor, Tensor],
        reset_mask: Optional[Tensor] = None,
        embed: Optional[Tensor] = None,
        temperature: float = 1.0,
    ):
        in_h, in_z = in_state
        if reset_mask is not None:
            in_h = in_h * reset_mask
            in_z = in_z * reset_mask

        B = action.shape[0]

        # concat in original dreamerv2, added in pydreamer
        x = self.z_mlp(in_z) + self.a_mlp(action)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h)

        if embed is not None:
            # concat in original dreamerv2, added in pydreamer
            x = self.post_mlp_h(h) + self.post_mlp_e(embed)
            norm_layer, mlp = self.post_norm, self.post_mlp
        else:
            x = self.prior_mlp_h(h)
            norm_layer, mlp = self.prior_norm, self.prior_mlp

        x = norm_layer(x)
        x = F.elu(x)
        pp = mlp(x)  # posterior or prior
        distr = self.zdistr(pp, temperature)
        sample = distr.rsample().reshape(B, -1)

        return pp, (h, sample)

    def batch_prior(self, h: Tensor) -> Tensor:  # tensor(T, B, D)
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)  # tensor(B,2S)
        return prior

    def zdistr(self, pp: Tensor, temperature: float = 1.0) -> D.Distribution:
        # pp = posterior or prior
        logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_rank)) / temperature
        distr = D.OneHotCategoricalStraightThrough(logits=logits.float())
        distr = D.independent.Independent(distr, 1)
        return distr
