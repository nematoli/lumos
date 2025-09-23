from itertools import chain
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
        rgb_decoder: DictConfig,
        rssm: DictConfig,
        amp: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        train_batch_size: int,
        val_batch_size: int,
        with_proprio: bool,
        use_rgb_decoder: bool,
        use_gripper_camera: bool,
        robot_dim: int,
        name: str,
    ):
        super(DreamerV2, self).__init__(name=name)
        self.use_rgb_decoder = use_rgb_decoder
        self.use_gripper_camera = use_gripper_camera
        rssm.cell.embed_dim = encoder.out_dim*encoder.num_patches + robot_dim if with_proprio else encoder.out_dim*encoder.num_patches
        self.encoder = hydra.utils.instantiate(encoder)
        decoder.in_dim = rssm.cell.deter_dim + rssm.cell.stoch_dim * rssm.cell.stoch_rank
        self.decoder = hydra.utils.instantiate(decoder)
        if self.use_rgb_decoder:
            rgb_decoder.use_gripper_camera = self.use_gripper_camera
            rgb_decoder.in_dim = decoder.in_dim
            self.rgb_decoder = hydra.utils.instantiate(rgb_decoder)
            self.rgb_opt_scaler = hydra.utils.instantiate(amp.scaler)
        self.with_proprio = with_proprio
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
        self.patch_weight = loss.patch_weight

        self.automatic_optimization = False

        self.batch_metrics = [
            "loss_total",
            "loss_patch",
            "loss_img",
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
        optimizer = hydra.utils.instantiate(
            self.optimizer,
            params=chain(self.encoder.parameters(), self.decoder.parameters(), self.rssm_core.parameters()),
        )
        if self.use_rgb_decoder:
            rgb_optimizer = hydra.utils.instantiate(self.optimizer, params=self.rgb_decoder.parameters())
        else:
            rgb_optimizer = None
        return [optimizer, rgb_optimizer]

    def forward(
        self,
        patches: Tensor,
        proprio: Tensor,
        act: Tensor,
        reset: Tensor,
        in_state: Tensor,
    ) -> Dict[str, Tensor]:
        embed = self.encoder(patches)
        if self.with_proprio:
            embed = torch.cat((embed, proprio), -1)

        prior, post, features, out_states = self.rssm_core.forward(embed, act, reset, in_state)

        dcd_patches = self.decoder(features)
        if self.use_rgb_decoder:
            dcd_img_s, dcd_img_g = self.rgb_decoder(features.detach())
        else:
            dcd_img_s, dcd_img_g = None, None

        outputs = {
            "prior": prior,
            "post": post,
            "features": features,
            "dcd_patches": dcd_patches,
            "dcd_img_s": dcd_img_s,
            "dcd_img_g": dcd_img_g,
            "out_states": out_states,
        }

        return outputs

    @torch.inference_mode()
    def infer_features(
        self,
        patches: Tensor,
        proprio: Tensor,
        actions: Tensor,
        reset: Tensor,
        in_state: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        self.eval()
        with self.autocast:
            # Prepare arguments for self(), ensuring device placement
            patches_dev = patches.to(self.device)
            proprio_dev = proprio.to(self.device)
            actions_dev = actions.to(self.device)
            reset_dev = reset.to(self.device)

            outs = self(
                patches_dev,
                proprio_dev,
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

    def pred_img(self, prior: Tensor, features: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        with torch.no_grad():
            prior_samples = self.rssm_core.zdistr(prior).sample()
            prior_samples = prior_samples.reshape(prior_samples.shape[0], prior_samples.shape[1], -1)
            features_prior = self.rssm_core.feature_replace_z(features, prior_samples)
            dcd_img_s = self.rgb_decoder(features_prior)
            return dcd_img_s

    def on_train_epoch_start(self) -> None:
        super(DreamerV2, self).on_train_epoch_start()
        self.in_state = self.rssm_core.init_state(self.train_batch_size)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        opt, rgb_opt = self.optimizers()
        opt.zero_grad()
        batch = batch["vis"]

        with self.autocast:
            outs = self(
                batch["patches"],
                batch["robot_obs"],
                batch["actions"]["pre_actions"],
                batch["reset"],
                self.in_state,
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        self.log_metrics(losses, mode="train")
        if self.global_step % self.trainer.log_every_n_steps == 0:
            pred_img_s, pred_img_g = self.pred_img(*samples)

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
        torch.nn.utils.clip_grad_norm_(
            chain(self.encoder.parameters(), self.decoder.parameters(), self.rssm_core.parameters()), self.grad_clip
        )
        self.scaler.step(opt)
        self.scaler.update()

        if self.use_rgb_decoder:
            rgb_opt.zero_grad()
            self.rgb_opt_scaler.scale(losses["loss_img"]).backward()
            torch.nn.utils.clip_grad_norm_(self.rgb_decoder.parameters(), self.grad_clip)
            self.rgb_opt_scaler.step(rgb_opt)
            self.rgb_opt_scaler.update()

        return losses["loss_total"]

    def on_validation_epoch_start(self) -> None:
        super(DreamerV2, self).on_validation_epoch_start()
        self.in_state = self.rssm_core.init_state(self.val_batch_size)
        self.running_metrics = {metric_name: 0 for metric_name in self.batch_metrics}
        self.num_val_batches = 0

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        with self.autocast:
            outs = self(
                batch["patches"],
                batch["robot_obs"],
                batch["actions"]["pre_actions"],
                batch["reset"],
                self.in_state,
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
            self.val_gt_img_g = batch["rgb_obs"]["rgb_gripper"][-1, 0]
            self.val_dcd_img_g = outs["dcd_img_g"][-1, 0]

        self.val_samples = samples
        return losses["loss_total"]

    def on_validation_epoch_end(self) -> None:
        for key in self.running_metrics.keys():
            self.running_metrics[key] /= self.num_val_batches
        self.log_metrics(self.running_metrics, mode="val")
        pred_img_s, pred_img_g = self.pred_img(*self.val_samples)

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

        loss_reconstr = 0.5 * torch.square(outs["dcd_patches"] - batch["patches"]).sum(dim=[-1, -2, -3])  # MSE
        loss = self.kl_weight * loss_kl + self.patch_weight * loss_reconstr

        if self.use_rgb_decoder:
            obs_list = [batch["rgb_obs"]["rgb_static"]]
            dcd_img_list = [outs["dcd_img_s"]]

            if self.use_gripper_camera:
                obs_list.append(batch["rgb_obs"].get("rgb_gripper"))
                dcd_img_list.append(outs.get("dcd_img_g"))

            obs = torch.cat(obs_list, dim=2)
            dcd_img = torch.cat(dcd_img_list, dim=2)
            loss_img_reconstr = 0.5 * torch.square(dcd_img - obs).sum(dim=[-1, -2, -3])  # MSE

        metrics = {
            "loss_total": loss,
            "loss_patch": loss_reconstr,
            "loss_img": loss_img_reconstr,
            "loss_kl": loss_kl,
            "loss_kl-post": loss_kl_post,
            "loss_kl-prior": loss_kl_prior,
            "entropy_prior": dprior.entropy(),
            "entropy_post": dpost.entropy(),
        }

        metrics = {k: v.mean() for k, v in metrics.items()}

        return metrics
