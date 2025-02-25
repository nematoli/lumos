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

from lumos.utils.calvin_lowdim_rot6d import CALVINLowdimWrapper
import wandb
import yaml
from lumos.utils.transforms import NormalizeVector, NormalizeVectorMinMax, UnnormalizeVectorMinMax

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


class DreamerV2_Hybrid(WorldModel):
    """
    The lightning module used for training DreamerV2.
    Args:
    """

    def __init__(
        self,
        encoder: DictConfig,
        encoder_state: DictConfig,
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
        super(DreamerV2_Hybrid, self).__init__(name=name)

        self.rgb_encoder = hydra.utils.instantiate(encoder)
        self.state_encoder = hydra.utils.instantiate(encoder_state)
        decoder.in_dim = rssm.cell.deter_dim + rssm.cell.stoch_dim * rssm.cell.stoch_rank
        self.decoder = hydra.utils.instantiate(decoder)
        rssm.cell.embed_dim = encoder.cnn_depth * 32 + encoder_state.output_dim
        self.with_proprio = with_proprio
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
        self.state_weight = loss.state_weight
        self.grad_clip = loss.grad_clip

        self.automatic_optimization = False

        self.batch_metrics = [
            "loss_total",
            "loss_img",
            "loss_state",
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

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        return {"optimizer": optimizer}

    def forward(
        self,
        rgb_s: Tensor,
        rgb_g: Tensor,
        scene_obs: Tensor,
        act: Tensor,
        reset: Tensor,
        in_state: Tensor,
    ) -> Dict[str, Tensor]:

        rgb_embed = self.rgb_encoder(rgb_s, rgb_g)
        state_embed = self.state_encoder(scene_obs)
        embed = torch.cat([rgb_embed, state_embed], dim=-1)
        # if self.with_proprio:
        #     embed = torch.cat((embed, proprio), -1)

        # if local_act is not None:
        #     act = local_act
        # elif self.gripper_control:
        #     act = world_to_tcp_frame(act, robot_obs)

        prior, post, features, out_states = self.rssm_core.forward(embed, act, reset, in_state)

        dcd_img_s, dcd_img_g, dcd_scene_obs = self.decoder(features)

        outputs = {
            "prior": prior,
            "post": post,
            "features": features,
            "dcd_img_s": dcd_img_s,
            "dcd_img_g": dcd_img_g,
            "dcd_scene_obs": dcd_scene_obs,
            "out_states": out_states,
        }

        return outputs

    @torch.inference_mode()
    def infer_features(
        self,
        rgb_s: Tensor,
        rgb_g: Tensor,
        scene_obs: Tensor,
        actions: Tensor,
        reset: Tensor,
        in_state: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        self.eval()
        with self.autocast:
            outs = self(
                rgb_s.to(self.device),
                rgb_g.to(self.device),
                scene_obs.to(self.device),
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
            dcd_img_s, dcd_img_g, dcd_scene_obs = self.decoder(features_prior)
            return dcd_img_s, dcd_img_g, dcd_scene_obs

    def on_train_epoch_start(self) -> None:
        super(DreamerV2_Hybrid, self).on_train_epoch_start()
        self.in_state = self.rssm_core.init_state(self.batch_size)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:

        opt = self.optimizers()
        opt.zero_grad()

        with self.autocast:
            outs = self(
                batch["hybrid"]["rgb_obs"]["rgb_static"],
                batch["hybrid"]["rgb_obs"]["rgb_gripper"],
                batch["hybrid"]["scene_obs"],
                batch["hybrid"]["actions"]["pre_actions"],
                batch["hybrid"]["reset"],
                self.in_state,
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        self.log_metrics(losses, mode="train")
        if self.global_step % self.trainer.log_every_n_steps == 0:
            pred_img_s, pred_img_g, pred_scene_obs = self.pred_img(*samples)
            self.log_images(
                batch["hybrid"]["rgb_obs"]["rgb_static"][-1, 0],
                batch["hybrid"]["rgb_obs"]["rgb_gripper"][-1, 0],
                outs["dcd_img_s"][-1, 0],
                outs["dcd_img_g"][-1, 0],
                pred_img_s[-1, 0],
                pred_img_g[-1, 0],
                mode="train-rgb",
            )

            # Log images by rendering the environment from the state
            self.spawn_env()
            state_img_s, state_img_g = self.render_environment(batch["hybrid"]["scene_obs"][-1, 0])
            dcd_state_img_s, dcd_state_img_g = self.render_environment(outs["dcd_scene_obs"][-1, 0])
            pre_state_img_s, pre_state_img_g = self.render_environment(pred_scene_obs[-1, 0])
            self.log_images_state(
                state_img_s,
                state_img_g,
                dcd_state_img_s,
                dcd_state_img_g,
                pre_state_img_s,
                pre_state_img_g,
                mode="train-state",
            )

        self.scaler.scale(losses["loss_total"]).backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.scaler.step(opt)
        self.scaler.update()

        return losses["loss_total"]

    def on_validation_epoch_start(self) -> None:
        super(DreamerV2_Hybrid, self).on_validation_epoch_start()
        self.in_state = self.rssm_core.init_state(self.batch_size)
        self.running_metrics = {metric_name: 0 for metric_name in self.batch_metrics}

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        with self.autocast:
            outs = self(
                batch["hybrid"]["rgb_obs"]["rgb_static"],
                batch["hybrid"]["rgb_obs"]["rgb_gripper"],
                batch["hybrid"]["scene_obs"],
                batch["hybrid"]["actions"]["pre_actions"],
                batch["hybrid"]["reset"],
                self.in_state,
            )
            losses = self.loss(batch, outs)
            samples = (outs["prior"], outs["features"])

        self.in_state = outs["out_states"]

        for key in losses.keys():
            self.running_metrics[key] += losses[key]

        # keep track of last batch for logging
        self.val_gt_img_s = batch["hybrid"]["rgb_obs"]["rgb_static"][-1, 0]
        self.val_gt_img_g = batch["hybrid"]["rgb_obs"]["rgb_gripper"][-1, 0]
        self.val_dcd_img_s = outs["dcd_img_s"][-1, 0]
        self.val_dcd_img_g = outs["dcd_img_g"][-1, 0]
        self.val_samples = samples

        self.val_state_img_s, self.val_state_img_g = self.render_environment(batch["hybrid"]["scene_obs"][-1, 0])
        self.val_dcd_state_img_s, self.val_dcd_state_img_g = self.render_environment(outs["dcd_scene_obs"][-1, 0])
        return losses["loss_total"]

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            num_val_batches = 2
        else:
            num_val_batches = len(self.trainer.datamodule.val_dataloader())
        for key in self.running_metrics.keys():
            self.running_metrics[key] /= num_val_batches
        self.log_metrics(self.running_metrics, mode="val")
        pred_img_s, pred_img_g, pred_scene_obs = self.pred_img(*self.val_samples)
        self.log_images(
            self.val_gt_img_s,
            self.val_gt_img_g,
            self.val_dcd_img_s,
            self.val_dcd_img_g,
            pred_img_s[-1, 0],
            pred_img_g[-1, 0],
            mode="val-rgb",
        )

        # Log images by rendering the environment from the state
        pred_state_img_s, pred_state_img_g = self.render_environment(pred_scene_obs[-1, 0])
        self.log_images_state(
            self.val_state_img_s,
            self.val_state_img_g,
            self.val_dcd_state_img_s,
            self.val_dcd_state_img_g,
            pred_state_img_s,
            pred_state_img_g,
            mode="val-state",
        )

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        dpost = self.rssm_core.zdistr(outs["post"])
        dprior = self.rssm_core.zdistr(outs["prior"])
        loss_kl_post = D.kl.kl_divergence(dpost, self.rssm_core.zdistr(outs["prior"].detach()))
        loss_kl_prior = D.kl.kl_divergence(self.rssm_core.zdistr(outs["post"].detach()), dprior)
        loss_kl = (1 - self.kl_balance) * loss_kl_post + self.kl_balance * loss_kl_prior

        obs = torch.cat([batch["hybrid"]["rgb_obs"]["rgb_static"], batch["hybrid"]["rgb_obs"]["rgb_gripper"]], dim=2)
        dcd_img = torch.cat([outs["dcd_img_s"], outs["dcd_img_g"]], dim=2)
        loss_reconstr = 0.5 * torch.square(dcd_img - obs).sum(dim=[-1, -2, -3])  # MSE
        loss_state_reconstr = 0.5 * torch.square(outs["dcd_scene_obs"] - batch["hybrid"]["scene_obs"]).sum(
            dim=[-1, -2, -3]
        )
        loss = self.kl_weight * loss_kl + self.image_weight * loss_reconstr + self.state_weight * loss_state_reconstr

        metrics = {
            "loss_total": loss,
            "loss_img": loss_reconstr,
            "loss_state": loss_state_reconstr,
            "loss_kl": loss_kl,
            "loss_kl-post": loss_kl_post,
            "loss_kl-prior": loss_kl_prior,
            "entropy_prior": dprior.entropy(),
            "entropy_post": dpost.entropy(),
        }

        metrics = {k: v.mean() for k, v in metrics.items()}

        return metrics

    def spawn_env(self):
        if self.env is None:
            self.env = CALVINLowdimWrapper(
                self.calvin_env_cfg,
                render_hw=(200, 200),
                render_cams=["rgb_static", "rgb_gripper"],
                device=f"{self.device.type}:{self.device.index}",
            )

    def render_environment(self, state):
        return self.render_environment_minmax(state)
        # return self.render_environment_meanstd(state)

    def render_environment_minmax(self, state):
        stats_path = "/home/lagandua/projects/dppo/data/statistics_minmax.yaml"
        with open(stats_path, "r") as f:
            stats = yaml.safe_load(f)
            robot_obs_min = torch.tensor(stats["robot_obs"][0]["min"])
            robot_obs_max = torch.tensor(stats["robot_obs"][0]["max"])
            scene_obs_min = torch.tensor(stats["scene_obs"][0]["min"])
            scene_obs_max = torch.tensor(stats["scene_obs"][0]["max"])

        # Unnormalize the state
        unnormalize_robot = UnnormalizeVectorMinMax(robot_obs_min, robot_obs_max)
        unnormalize_scene = UnnormalizeVectorMinMax(scene_obs_min, scene_obs_max)

        if len(state) == 33:
            scene_obs = unnormalize_scene(state.detach().cpu()).numpy()
            robot_obs = None
        else:
            scene_obs = unnormalize_scene(state[18:].detach().cpu()).numpy()
            robot_obs = unnormalize_robot(state[:18].detach().cpu()).numpy()

        scene_obs[4] = 0 if scene_obs[4] < 1 else 1
        scene_obs[5] = 0 if scene_obs[5] < 1 else 1
        self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
        renders = self.env.render()
        return renders[0], renders[1]

    def render_environment_meanstd(self, state):
        stats_path = "/home/lagandua/projects/dppo/data/statistics_old.yaml"
        with open(stats_path, "r") as f:
            stats = yaml.safe_load(f)
            robot_obs_mean = torch.tensor(stats["robot_obs"][0]["mean"])
            robot_obs_std = torch.tensor(stats["robot_obs"][0]["std"])
            scene_obs_mean = torch.tensor(stats["scene_obs"][0]["mean"])
            scene_obs_std = torch.tensor(stats["scene_obs"][0]["std"])

        # Unnormalize the state
        unnormalize_robot = NormalizeVector(-robot_obs_mean / robot_obs_std, 1 / robot_obs_std)
        unnormalize_scene = NormalizeVector(-scene_obs_mean / scene_obs_std, 1 / scene_obs_std)

        robot_obs = unnormalize_robot(state[:18].detach().cpu()).numpy()
        scene_obs = unnormalize_scene(state[18:].detach().cpu()).numpy()

        scene_obs[4] = 0 if scene_obs[4] < 1 else 1
        scene_obs[5] = 0 if scene_obs[5] < 1 else 1
        self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
        renders = self.env.render()
        return renders[0], renders[1]

    @torch.no_grad()
    def log_images_state(
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
