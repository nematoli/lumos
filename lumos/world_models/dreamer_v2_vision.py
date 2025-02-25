import logging
import hydra
from omegaconf import DictConfig
from typing import Dict, Union, Any, Tuple
import torch
from torch import Tensor
from pytorch_lightning.utilities import rank_zero_only
from lumos.utils.nn_utils import init_weights
from lumos.world_models.world_model import WorldModel
import torch.distributions as D
from lumos.utils.gripper_control import world_to_tcp_frame

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
        batch_size: int,
        with_proprio: bool,
        gripper_control: bool,
        name: str,
    ):
        super(DreamerV2, self).__init__(name=name)

        self.encoder = hydra.utils.instantiate(encoder)
        decoder.in_dim = rssm.cell.deter_dim + rssm.cell.stoch_dim * rssm.cell.stoch_rank
        self.decoder = hydra.utils.instantiate(decoder)
        rssm.cell.embed_dim = encoder.cnn_depth * 32  # * 2
        self.with_proprio = with_proprio
        if self.with_proprio:
            rssm.cell.embed_dim += 18
        self.rssm_core = hydra.utils.instantiate(rssm)
        self.autocast = hydra.utils.instantiate(amp.autocast)
        self.scaler = hydra.utils.instantiate(amp.scaler)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.gripper_control = gripper_control

        self.kl_balance = loss.kl_balance
        self.kl_weight = loss.kl_weight
        self.image_weight = loss.image_weight
        self.grad_clip = loss.grad_clip

        self.automatic_optimization = False

        self.batch_metrics = [
            "loss_total",
            "loss_img",
            "loss_kl",
            "loss_kl-post",
            "loss_kl-prior",
            "entropy_prior",
            "entropy_post",
        ]

        for m in self.modules():
            init_weights(m)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        return {"optimizer": optimizer}

    def forward(
        self,
        rgb_s: Tensor,
        rgb_g: Tensor,
        robot_obs: Tensor,
        act: Tensor,
        reset: Tensor,
        in_state: Tensor,
    ) -> Dict[str, Tensor]:

        embed = self.encoder(rgb_s, rgb_g)
        if self.with_proprio:
            embed = torch.cat((embed, robot_obs), -1)

        # if local_act is not None:
        #     act = local_act
        # elif self.gripper_control:
        #     act = world_to_tcp_frame(act, robot_obs)

        prior, post, features, out_states = self.rssm_core.forward(embed, act, reset, in_state)

        dcd_img_s, dcd_img_g = self.decoder(features)

        outputs = {
            "prior": prior,
            "post": post,
            "features": features,
            "dcd_img_s": dcd_img_s,
            "dcd_img_g": dcd_img_g,
            "out_states": out_states,
        }

        return outputs

    @torch.inference_mode()
    def infer_features(
        self,
        rgb_s: Tensor,
        rgb_g: Tensor,
        robot_obs: Tensor,
        actions: Tensor,
        reset: Tensor,
        in_state: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        self.eval()
        with self.autocast:
            outs = self(
                rgb_s.to(self.device),
                rgb_g.to(self.device),
                robot_obs.to(self.device),
                actions.to(self.device),
                reset.to(self.device),
                in_state,
            )

        return outs["features"], outs["out_states"]

    def dream(self, act: Tensor, in_state: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor]:
        with self.autocast:
            pp, (h, z) = self.rssm_core.cell.forward(act, in_state, temperature=temperature)
        return pp, (h, z)

    def pred_img(self, prior: Tensor, features: Tensor) -> Tensor:
        with torch.no_grad():
            prior_samples = self.rssm_core.zdistr(prior).sample()
            prior_samples = prior_samples.reshape(prior_samples.shape[0], prior_samples.shape[1], -1)
            features_prior = self.rssm_core.feature_replace_z(features, prior_samples)
            dcd_img_s, dcd_img_g = self.decoder(features_prior)
            return dcd_img_s, dcd_img_g

    def on_train_epoch_start(self) -> None:
        super(DreamerV2, self).on_train_epoch_start()
        self.in_state = self.rssm_core.init_state(self.batch_size)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:

        opt = self.optimizers()
        opt.zero_grad()

        with self.autocast:
            outs = self(
                batch["vis"]["rgb_obs"]["rgb_static"],
                batch["vis"]["rgb_obs"]["rgb_gripper"],
                batch["vis"]["robot_obs"],
                batch["vis"]["actions"]["pre_actions"],
                batch["vis"]["reset"],
                self.in_state,
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        self.log_metrics(losses, mode="train")
        if self.global_step % self.trainer.log_every_n_steps == 0:
            pred_img_s, pred_img_g = self.pred_img(*samples)
            self.log_images(
                batch["vis"]["rgb_obs"]["rgb_static"][-1, 0],
                batch["vis"]["rgb_obs"]["rgb_gripper"][-1, 0],
                outs["dcd_img_s"][-1, 0],
                outs["dcd_img_g"][-1, 0],
                pred_img_s[-1, 0],
                pred_img_g[-1, 0],
                mode="train-rgb",
            )

        self.scaler.scale(losses["loss_total"]).backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.scaler.step(opt)
        self.scaler.update()

        return losses["loss_total"]

    def on_validation_epoch_start(self) -> None:
        super(DreamerV2, self).on_validation_epoch_start()
        self.in_state = self.rssm_core.init_state(self.batch_size)
        self.running_metrics = {metric_name: 0 for metric_name in self.batch_metrics}

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        with self.autocast:
            outs = self(
                batch["vis"]["rgb_obs"]["rgb_static"],
                batch["vis"]["rgb_obs"]["rgb_gripper"],
                batch["vis"]["robot_obs"],
                batch["vis"]["actions"]["pre_actions"],
                batch["vis"]["state_info"]["pre_robot_obs"],
                batch["vis"]["reset"],
                self.in_state,
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        for key in losses.keys():
            self.running_metrics[key] += losses[key]

        # keep track of last batch for logging
        self.val_gt_img_s = batch["vis"]["rgb_obs"]["rgb_static"][-1, 0]
        self.val_gt_img_g = batch["vis"]["rgb_obs"]["rgb_gripper"][-1, 0]
        self.val_dcd_img_s = outs["dcd_img_s"][-1, 0]
        self.val_dcd_img_g = outs["dcd_img_g"][-1, 0]
        self.val_samples = samples
        return losses["loss_total"]

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            num_val_batches = 2
        else:
            num_val_batches = len(self.trainer.datamodule.val_dataloader())
        for key in self.running_metrics.keys():
            self.running_metrics[key] /= num_val_batches
        self.log_metrics(self.running_metrics, mode="val")
        pred_img_s, pred_img_g = self.pred_img(*self.val_samples)
        self.log_images(
            self.val_gt_img_s,
            self.val_gt_img_g,
            self.val_dcd_img_s,
            self.val_dcd_img_g,
            pred_img_s[-1, 0],
            pred_img_g[-1, 0],
            mode="val-rgb",
        )

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        dpost = self.rssm_core.zdistr(outs["post"])
        dprior = self.rssm_core.zdistr(outs["prior"])
        loss_kl_post = D.kl.kl_divergence(dpost, self.rssm_core.zdistr(outs["prior"].detach()))
        loss_kl_prior = D.kl.kl_divergence(self.rssm_core.zdistr(outs["post"].detach()), dprior)
        loss_kl = (1 - self.kl_balance) * loss_kl_post + self.kl_balance * loss_kl_prior

        obs = torch.cat([batch["vis"]["rgb_obs"]["rgb_static"], batch["vis"]["rgb_obs"]["rgb_gripper"]], dim=2)
        dcd_img = torch.cat([outs["dcd_img_s"], outs["dcd_img_g"]], dim=2)
        loss_reconstr = 0.5 * torch.square(dcd_img - obs).sum(dim=[-1, -2, -3])  # MSE

        loss = self.kl_weight * loss_kl + self.image_weight * loss_reconstr

        metrics = {
            "loss_total": loss,
            "loss_img": loss_reconstr,
            "loss_kl": loss_kl,
            "loss_kl-post": loss_kl_post,
            "loss_kl-prior": loss_kl_prior,
            "entropy_prior": dprior.entropy(),
            "entropy_post": dpost.entropy(),
        }

        metrics = {k: v.mean() for k, v in metrics.items()}

        return metrics
