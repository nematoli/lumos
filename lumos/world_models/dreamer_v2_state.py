import logging
from typing import Any, Dict, Tuple, Union

import hydra
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch import Tensor
import torch.distributions as D

from lumos.utils.nn_utils import init_weights
from lumos.world_models.world_model import WorldModel

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


class DreamerV2(WorldModel):
    """
    The lightning module used for training DreamerV2.
    Args:
    """

    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        rssm: DictConfig,
        amp: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        train_batch_size: int,
        val_batch_size: int,
        name: str,
    ):
        super(DreamerV2, self).__init__(name=name)

        self.encoder = hydra.utils.instantiate(encoder)
        decoder.in_dim = rssm.cell.deter_dim + rssm.cell.stoch_dim * rssm.cell.stoch_rank
        self.decoder = hydra.utils.instantiate(decoder)
        rssm.cell.embed_dim = encoder.output_dim

        self.rssm_core = hydra.utils.instantiate(rssm)
        self.autocast = hydra.utils.instantiate(amp.autocast)
        self.scaler = hydra.utils.instantiate(amp.scaler)
        self.optimizer = optimizer
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.kl_balance = loss.kl_balance
        self.kl_weight = loss.kl_weight
        self.image_weight = loss.image_weight
        self.grad_clip = loss.grad_clip

        self.automatic_optimization = False

        self.batch_metrics = [
            "loss_total",
            "loss_recon",
            "loss_kl",
            "loss_kl-post",
            "loss_kl-prior",
            "entropy_prior",
            "entropy_post",
        ]

        for m in self.modules():
            init_weights(m)
        self.num_val_batches = 0
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        return {"optimizer": optimizer}

    def forward(
        self,
        s_obs: Tensor,
        act: Tensor,
        reset: Tensor,
        in_state: Tensor,
    ) -> Dict[str, Tensor]:
        embed = self.encoder(s_obs)

        prior, post, features, out_states = self.rssm_core.forward(embed, act, reset, in_state)

        dcd_s_obs = self.decoder(features)

        outputs = {
            "prior": prior,
            "post": post,
            "features": features,
            "dcd_s_obs": dcd_s_obs,
            "out_states": out_states,
        }

        return outputs

    @torch.inference_mode()
    def infer_features(
        self,
        s_obs: Tensor,
        actions: Tensor,
        reset: Tensor,
        in_state: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        self.eval()
        with self.autocast:
            # Prepare arguments for self(), ensuring device placement
            s_obs_dev = s_obs.to(self.device)
            actions_dev = actions.to(self.device)
            reset_dev = reset.to(self.device)

            outs = self(
                s_obs_dev,
                actions_dev,
                reset_dev,
                in_state,
            )

        # features = torch.cat((outs["features"], outs["prior"]), -1)
        return outs["features"], outs["out_states"]

    def dream(self, act: Tensor, in_state: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor]:
        with self.autocast:
            pp, (h, z) = self.rssm_core.cell.forward(act, in_state, temperature=temperature)
        return pp, (h, z)

    def on_train_epoch_start(self) -> None:
        super(DreamerV2, self).on_train_epoch_start()
        self.in_state = self.rssm_core.init_state(self.train_batch_size)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        opt = self.optimizers()
        opt.zero_grad()
        batch = batch["state"]

        with self.autocast:
            outs = self(
                batch["state_obs"],
                batch["actions"]["pre_actions"],
                batch["reset"],
                self.in_state,
            )
            losses = self.loss(batch, outs)

        self.in_state = outs["out_states"]

        self.log_metrics(losses, mode="train")

        self.scaler.scale(losses["loss_total"]).backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.scaler.step(opt)
        self.scaler.update()

        return losses["loss_total"]

    def on_validation_epoch_start(self) -> None:
        super(DreamerV2, self).on_validation_epoch_start()
        self.in_state = self.rssm_core.init_state(self.val_batch_size)
        self.running_metrics = {metric_name: 0 for metric_name in self.batch_metrics}
        self.num_val_batches = 0

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        with self.autocast:
            outs = self(
                batch["state_obs"],
                batch["actions"]["pre_actions"],
                batch["reset"],
                self.in_state,
            )
            losses = self.loss(batch, outs)

        self.in_state = outs["out_states"]

        for key in losses.keys():
            self.running_metrics[key] += losses[key]
        self.num_val_batches += 1

        # keep track of last batch for logging
        return losses["loss_total"]

    def on_validation_epoch_end(self) -> None:
        for key in self.running_metrics.keys():
            self.running_metrics[key] /= self.num_val_batches
        self.log_metrics(self.running_metrics, mode="val")

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        dpost = self.rssm_core.zdistr(outs["post"])
        dprior = self.rssm_core.zdistr(outs["prior"])
        loss_kl_post = D.kl.kl_divergence(dpost, self.rssm_core.zdistr(outs["prior"].detach()))
        loss_kl_prior = D.kl.kl_divergence(self.rssm_core.zdistr(outs["post"].detach()), dprior)
        loss_kl = (1 - self.kl_balance) * loss_kl_post + self.kl_balance * loss_kl_prior

        obs = batch["state_obs"]
        dcd_s_obs = outs["dcd_s_obs"]
        loss_reconstr = 0.5 * torch.square(dcd_s_obs - obs).sum(dim=[-1, -2, -3])  # MSE

        loss = self.kl_weight * loss_kl + self.image_weight * loss_reconstr

        metrics = {
            "loss_total": loss,
            "loss_recon": loss_reconstr,
            "loss_kl": loss_kl,
            "loss_kl-post": loss_kl_post,
            "loss_kl-prior": loss_kl_prior,
            "entropy_prior": dprior.entropy(),
            "entropy_post": dpost.entropy(),
        }

        metrics = {k: v.mean() for k, v in metrics.items()}

        return metrics
