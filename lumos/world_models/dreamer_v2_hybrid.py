import logging
from typing import Any, Dict, Optional, Tuple, Union

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
        use_gripper_camera: bool,
        name: str,
    ):
        super(DreamerV2, self).__init__(name=name)
        self.use_gripper_camera = use_gripper_camera

        self.encoder = hydra.utils.instantiate(encoder)
        decoder.in_dim = rssm.cell.deter_dim + rssm.cell.stoch_dim * rssm.cell.stoch_rank
        decoder.use_gripper_camera = self.use_gripper_camera
        self.decoder = hydra.utils.instantiate(decoder)
        rssm.cell.embed_dim = encoder.cnn_depth * 32 + encoder.state_out_dim
        self.rssm_core = hydra.utils.instantiate(rssm)
        self.autocast = hydra.utils.instantiate(amp.autocast)
        self.scaler = hydra.utils.instantiate(amp.scaler)
        self.optimizer = optimizer
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.kl_balance = loss.kl_balance
        self.kl_weight = loss.kl_weight
        self.image_weight = loss.image_weight
        self.state_weight = loss.state_weight
        self.grad_clip = loss.grad_clip

        self.automatic_optimization = False

        self.batch_metrics = [
            "loss_total",
            "loss_img",
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
        rgb_s: Tensor,
        s_obs: Tensor,
        act: Tensor,
        reset: Tensor,
        in_state: Tensor,
        rgb_g: Tensor = None,
    ) -> Dict[str, Tensor]:
        embed = self.encoder(rgb_s, rgb_g, s_obs)

        prior, post, features, out_states = self.rssm_core.forward(embed, act, reset, in_state)

        dcd_img_s, dcd_img_g, dcd_s_obs = self.decoder(features)

        outputs = {
            "prior": prior,
            "post": post,
            "features": features,
            "dcd_img_s": dcd_img_s,
            "dcd_img_g": dcd_img_g,
            "dcd_s_obs": dcd_s_obs,
            "out_states": out_states,
        }

        return outputs

    @torch.inference_mode()
    def infer_features(
        self,
        rgb_s: Tensor,
        s_obs: Tensor,
        actions: Tensor,
        reset: Tensor,
        in_state: Tensor,
        rgb_g: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        self.eval()
        with self.autocast:
            # Prepare arguments for self(), ensuring device placement
            rgb_s_dev = rgb_s.to(self.device)
            rgb_g_dev = rgb_g.to(self.device) if rgb_g is not None else None
            s_obs_dev = s_obs.to(self.device)
            actions_dev = actions.to(self.device)
            reset_dev = reset.to(self.device)

            outs = self(
                rgb_s_dev,
                s_obs_dev,
                actions_dev,
                reset_dev,
                in_state,
                rgb_g=rgb_g_dev,
            )

        # features = torch.cat((outs["features"], outs["prior"]), -1)
        return outs["features"], outs["out_states"]

    def dream(self, act: Tensor, in_state: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor]:
        with self.autocast:
            pp, (h, z) = self.rssm_core.cell.forward(act, in_state, temperature=temperature)
        return pp, (h, z)

    def pred_img_s(self, prior: Tensor, features: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        with torch.no_grad():
            prior_samples = self.rssm_core.zdistr(prior).sample()
            prior_samples = prior_samples.reshape(prior_samples.shape[0], prior_samples.shape[1], -1)
            features_prior = self.rssm_core.feature_replace_z(features, prior_samples)
            dcd_img_s, dcd_img_g, dcd_s_obs = self.decoder(features_prior)
            return dcd_img_s, dcd_img_g, dcd_s_obs

    def on_train_epoch_start(self) -> None:
        super(DreamerV2, self).on_train_epoch_start()
        self.in_state = self.rssm_core.init_state(self.train_batch_size)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        opt = self.optimizers()
        opt.zero_grad()
        batch = batch["hybrid"]

        with self.autocast:
            rgb_g_input = batch["rgb_obs"].get("rgb_gripper")
            outs = self(
                batch["rgb_obs"]["rgb_static"],
                batch["state_obs"],
                batch["actions"]["pre_actions"],
                batch["reset"],
                self.in_state,
                rgb_g=rgb_g_input,
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        self.log_metrics(losses, mode="train")
        if self.global_step % self.trainer.log_every_n_steps == 0:
            pred_img_s, pred_img_g, _ = self.pred_img_s(*samples)

            gt_img_g_val, dcd_img_g_val, pred_img_g_val = None, None, None
            if self.use_gripper_camera:
                gt_img_g_val = batch["rgb_obs"].get("rgb_gripper")[-1, 0]
                dcd_img_g_val = outs.get("dcd_img_g")[-1, 0]
                pred_img_g_val = pred_img_g[-1, 0]

            self.log_images(
                gt_img_s=batch["rgb_obs"]["rgb_static"][-1, 0],
                dcd_img_s=outs["dcd_img_s"][-1, 0],
                pred_img_s=pred_img_s[-1, 0],
                mode="train",
                gt_img_g=gt_img_g_val,
                dcd_img_g=dcd_img_g_val,
                pred_img_g=pred_img_g_val,
            )

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
            rgb_g_input = batch["rgb_obs"].get("rgb_gripper")
            outs = self(
                batch["rgb_obs"]["rgb_static"],
                batch["state_obs"],
                batch["actions"]["pre_actions"],
                batch["reset"],
                self.in_state,
                rgb_g=rgb_g_input,
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        for key in losses.keys():
            self.running_metrics[key] += losses[key]
        self.num_val_batches += 1

        # keep track of last batch for logging
        self.val_gt_img_s = batch["rgb_obs"]["rgb_static"][-1, 0]
        self.val_dcd_img_s = outs["dcd_img_s"][-1, 0]

        self.val_gt_img_g, self.val_dcd_img_g = None, None
        if self.use_gripper_camera:
            self.val_gt_img_g = rgb_g_input[-1, 0]
            self.val_dcd_img_g = outs.get("dcd_img_g")[-1, 0]

        self.val_samples = samples
        return losses["loss_total"]

    def on_validation_epoch_end(self) -> None:
        for key in self.running_metrics.keys():
            self.running_metrics[key] /= self.num_val_batches
        self.log_metrics(self.running_metrics, mode="val")
        pred_img_s, pred_img_g, _ = self.pred_img_s(*self.val_samples)

        pred_img_g_val = None
        if self.use_gripper_camera:
            pred_img_g_val = pred_img_g[-1, 0]

        self.log_images(
            gt_img_s=self.val_gt_img_s,
            dcd_img_s=self.val_dcd_img_s,
            pred_img_s=pred_img_s[-1, 0],
            mode="val",
            gt_img_g=self.val_gt_img_g,
            dcd_img_g=self.val_dcd_img_g,
            pred_img_g=pred_img_g_val,
        )

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        dpost = self.rssm_core.zdistr(outs["post"])
        dprior = self.rssm_core.zdistr(outs["prior"])
        loss_kl_post = D.kl.kl_divergence(dpost, self.rssm_core.zdistr(outs["prior"].detach()))
        loss_kl_prior = D.kl.kl_divergence(self.rssm_core.zdistr(outs["post"].detach()), dprior)
        loss_kl = (1 - self.kl_balance) * loss_kl_post + self.kl_balance * loss_kl_prior

        obs_list = [batch["rgb_obs"]["rgb_static"]]
        dcd_img_list = [outs["dcd_img_s"]]

        if self.use_gripper_camera:
            obs_list.append(batch["rgb_obs"].get("rgb_gripper"))
            dcd_img_list.append(outs.get("dcd_img_g"))

        obs = torch.cat(obs_list, dim=2)
        dcd_img = torch.cat(dcd_img_list, dim=2)
        loss_reconstr = 0.5 * torch.square(dcd_img - obs).sum(dim=[-1, -2, -3])  # MSE

        s_obs = batch["state_obs"]
        dcd_s_obs = outs["dcd_s_obs"]
        loss_state_reconstr = self.state_weight * 0.5 * torch.square(dcd_s_obs - s_obs).sum(dim=[-1, -2, -3])
        loss = self.kl_weight * loss_kl + self.image_weight * loss_reconstr + self.state_weight * loss_state_reconstr

        metrics = {
            "loss_total": loss,
            "loss_img": loss_reconstr,
            "loss_recon": loss_state_reconstr,
            "loss_kl": loss_kl,
            "loss_kl-post": loss_kl_post,
            "loss_kl-prior": loss_kl_prior,
            "entropy_prior": dprior.entropy(),
            "entropy_post": dpost.entropy(),
        }

        metrics = {k: v.mean() for k, v in metrics.items()}

        return metrics
