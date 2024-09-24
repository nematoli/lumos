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
from lumos.utils.calvin_lowdim import CALVINLowdimWrapper
import wandb
import yaml
from lumos.utils.transforms import NormalizeVector

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


class DreamerV2_State(WorldModel):
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
        super(DreamerV2_State, self).__init__(name=name)

        self.encoder = hydra.utils.instantiate(encoder)
        decoder.in_dim = rssm.cell.deter_dim + rssm.cell.stoch_dim * rssm.cell.stoch_rank
        self.decoder = hydra.utils.instantiate(decoder)
        rssm.cell.embed_dim = encoder.output_dim
        # self.with_proprio = with_proprio
        # if self.with_proprio:
        #     rssm.cell.embed_dim += 15
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
        self.calvin_env_cfg = None
        self.env = None

    def spawn_env(self):
        if self.env is None:
            self.env = CALVINLowdimWrapper(
                self.calvin_env_cfg,
                render_hw=(200, 200),
                render_cams=["rgb_static", "rgb_gripper"],
                device=f"{self.device.type}:{self.device.index}",
            )

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        return {"optimizer": optimizer}

    def forward(
        self,
        state_obs: Tensor,
        act: Tensor,
        pre_robot_obs: Tensor,
        reset: Tensor,
        in_state: Tensor,
        local_act: Tensor = None,
    ) -> Dict[str, Tensor]:

        embed = self.encoder(state_obs)
        # if self.with_proprio:
        #     embed = torch.cat((embed, proprio), -1)

        if local_act is not None:
            act = local_act
        elif self.gripper_control:
            act = world_to_tcp_frame(act, pre_robot_obs)

        prior, post, features, out_states = self.rssm_core.forward(embed, act, reset, in_state)

        dcd_robot_scene_obs = self.decoder(features)

        outputs = {
            "prior": prior,
            "post": post,
            "features": features,
            "dcd_robot_scene_obs": dcd_robot_scene_obs,
            "out_states": out_states,
        }

        return outputs

    @torch.inference_mode()
    def infer_features(
        self,
        state_obs: Tensor,
        actions: Tensor,
        pre_robot_obs: Tensor,
        reset: Tensor,
        in_state: Tensor,
        local_act: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        self.eval()
        with self.autocast:
            outs = self(
                state_obs.to(self.device),
                actions.to(self.device),
                pre_robot_obs.to(self.device),
                reset.to(self.device),
                in_state,
                local_act=local_act,
            )

        return outs["features"], outs["out_states"]

    def dream(self, act: Tensor, in_state: Tensor, temperature: float = 1.0) -> Tuple[Tensor, Tensor]:
        with self.autocast:
            pp, (h, z) = self.rssm_core.cell.forward(act, in_state, temperature=temperature)
        return pp, (h, z)

    def pred_state_obs(self, prior: Tensor, features: Tensor) -> Tensor:
        with torch.no_grad():
            prior_samples = self.rssm_core.zdistr(prior).sample()
            prior_samples = prior_samples.reshape(prior_samples.shape[0], prior_samples.shape[1], -1)
            features_prior = self.rssm_core.feature_replace_z(features, prior_samples)
            dcd_state_obs = self.decoder(features_prior)
            return dcd_state_obs

    def on_train_epoch_start(self) -> None:
        super(DreamerV2_State, self).on_train_epoch_start()
        self.in_state = self.rssm_core.init_state(self.batch_size)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:

        opt = self.optimizers()
        opt.zero_grad()

        with self.autocast:
            outs = self(
                batch["state"]["state_obs"],
                batch["state"]["actions"]["pre_actions"],
                batch["state"]["state_info"]["pre_robot_obs"],
                batch["state"]["reset"],
                self.in_state,
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        self.log_metrics(losses, mode="train")

        if self.global_step % self.trainer.log_every_n_steps == 0:
            self.spawn_env()
            true_img_s, true_img_g = self.render_environment(batch["state"]["state_obs"][-1, 0])
            dcd_img_s, dcd_img_g = self.render_environment(outs["dcd_robot_scene_obs"][-1, 0])
            pred_state_obs = self.pred_state_obs(*samples)
            pred_img_s, pred_img_g = self.render_environment(pred_state_obs[-1, 0])
            self.log_images(
                true_img_s,
                true_img_g,
                dcd_img_s,
                dcd_img_g,
                pred_img_s,
                pred_img_g,
                mode="train",
            )

        self.scaler.scale(losses["loss_total"]).backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.scaler.step(opt)
        self.scaler.update()

        return losses["loss_total"]

    def on_validation_epoch_start(self) -> None:
        super(DreamerV2_State, self).on_validation_epoch_start()
        self.in_state = self.rssm_core.init_state(self.batch_size)
        self.running_metrics = {metric_name: 0 for metric_name in self.batch_metrics}

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        with self.autocast:
            outs = self(
                batch["state"]["state_obs"],
                batch["state"]["rel_actions"]["pre_actions"],
                batch["state"]["state_info"]["pre_robot_obs"],
                batch["state"]["reset"],
                self.in_state,
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        for key in losses.keys():
            self.running_metrics[key] += losses[key]

        true_img_s, true_img_g = self.render_environment(batch["state"]["state_obs"][-1, 0])
        dcd_img_s, dcd_img_g = self.render_environment(outs["dcd_robot_scene_obs"][-1, 0])

        # keep track of last batch for logging
        self.val_gt_img_s = true_img_s
        self.val_gt_img_g = true_img_g
        self.val_dcd_img_s = dcd_img_s
        self.val_dcd_img_g = dcd_img_g
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
        pred_state_obs = self.pred_state_obs(self.val_samples)
        pred_img_s, pred_img_g = self.render_environment(pred_state_obs[-1, 0])
        self.log_images(
            self.val_gt_img_s,
            self.val_gt_img_g,
            self.val_dcd_img_s,
            self.val_dcd_img_g,
            pred_img_s,
            pred_img_g,
            mode="train",
        )

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        dpost = self.rssm_core.zdistr(outs["post"])
        dprior = self.rssm_core.zdistr(outs["prior"])
        loss_kl_post = D.kl.kl_divergence(dpost, self.rssm_core.zdistr(outs["prior"].detach()))
        loss_kl_prior = D.kl.kl_divergence(self.rssm_core.zdistr(outs["post"].detach()), dprior)
        loss_kl = (1 - self.kl_balance) * loss_kl_post + self.kl_balance * loss_kl_prior

        loss_reconstr = 0.5 * torch.square(outs["dcd_robot_scene_obs"] - batch["state"]["state_obs"]).sum(
            dim=[-1, -2, -3]
        )  # MSE

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

    def render_environment(self, state) -> None:
        with open("/home/lagandua/projects/lumos/dataset/calvin/statistics.yaml", "r") as f:
            stats = yaml.safe_load(f)
            robot_obs_mean = torch.tensor(stats["robot_obs"][0]["mean"])
            robot_obs_std = torch.tensor(stats["robot_obs"][0]["std"])
            scene_obs_mean = torch.tensor(stats["scene_obs"][0]["mean"])
            scene_obs_std = torch.tensor(stats["scene_obs"][0]["std"])

        # Unnormalize the state
        unnormalize_robot = NormalizeVector(-robot_obs_mean / robot_obs_std, 1 / robot_obs_std)
        unnormalize_scene = NormalizeVector(-scene_obs_mean / scene_obs_std, 1 / scene_obs_std)

        robot_obs = unnormalize_robot(state[:15].detach().cpu()).numpy()
        scene_obs = unnormalize_scene(state[15:].detach().cpu()).numpy()

        scene_obs[4] = 0 if scene_obs[4] < 1 else 1
        scene_obs[5] = 0 if scene_obs[5] < 1 else 1
        self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
        renders = self.env.render()
        return renders[0], renders[1]

    @torch.no_grad()
    def log_images(
        self,
        gt_img_s: Tensor,
        gt_img_g: Tensor,
        dcd_img_s: Tensor,
        dcd_img_g: Tensor,
        pred_img_s: Tensor,
        pred_img_g: Tensor,
        mode: str,
    ):
        """
        logs images
        Args:
            gt_obs (Tensor): Original image
            dcd_obs (Tensor): z reconstructed image
            pred_obs (Tensor): z-hat reconstructed image
        Returns:
            None
        """
        img1 = wandb.Image(gt_img_s, caption="gt")
        img2 = wandb.Image(dcd_img_s, caption="dcd")
        img3 = wandb.Image(pred_img_s, caption="pred")

        img4 = wandb.Image(gt_img_g, caption="gt_g")
        img5 = wandb.Image(dcd_img_g, caption="dcd_g")
        img6 = wandb.Image(pred_img_g, caption="pred_g")
        images = [img1, img2, img3, img4, img5, img6]

        wandb.log({f"imgs/{mode}": images})
