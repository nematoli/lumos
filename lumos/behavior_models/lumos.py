import logging
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import hydra
from omegaconf import DictConfig

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch import Tensor
import torch.nn.functional as F
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from transformers import CLIPProcessor, CLIPModel
from lumos.utils.rl_utils import advantage, max_cos, lambda_return, MC_return, action_mse
from lumos.utils.distributions import State
import torch.distributions as D
import torch.nn as nn
from torch.nn.functional import cross_entropy
from lumos.utils.gripper_control import tcp_to_world_frame, world_to_tcp_frame

from lumos.behavior_models.decoders.action_decoder import ActionDecoder

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


class LUMOS(pl.LightningModule, CalvinBaseModel):
    def __init__(
        self,
        perceptual_encoder: DictConfig,
        plan_proposal: DictConfig,
        plan_recognition: DictConfig,
        language_goal: DictConfig,
        visual_goal: DictConfig,
        action_decoder: DictConfig,
        critic: DictConfig,
        distribution: DictConfig,
        loss: DictConfig,
        actor_optimizer: DictConfig,
        critic_optimizer: DictConfig,
        world_model: pl.LightningModule,
        seq_len: int,
        name: str,
        use_clip_auxiliary_loss: bool,
        use_bc_loss: bool,
        temperature: float = 1.0,
        replan_freq: int = 30,
        gripper_control: bool = False,
        proj_vis_lang: Optional[DictConfig] = None,
    ) -> None:
        super(LUMOS, self).__init__()
        self.name = name
        self.wm = world_model
        self.seq_len = seq_len
        self.latent_half = world_model.rssm_core.cell.deter_dim

        self.setup_input_sizes(
            world_model,
            plan_proposal,
            plan_recognition,
            visual_goal,
            action_decoder,
            critic,
            distribution,
        )

        self.perceptual_encoder = hydra.utils.instantiate(perceptual_encoder)
        # plan networks
        self.dist = hydra.utils.instantiate(distribution)
        self.plan_proposal = hydra.utils.instantiate(plan_proposal, dist=self.dist)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition, dist=self.dist)

        # goal encoders
        self.visual_goal = hydra.utils.instantiate(visual_goal)
        self.language_goal = hydra.utils.instantiate(language_goal) if language_goal else None

        # actor and critic
        self.action_decoder: ActionDecoder = hydra.utils.instantiate(action_decoder)
        self.critic = hydra.utils.instantiate(critic)

        self.lambda_gae = loss.lambda_gae
        self.gamma = loss.gamma
        self.rho = loss.rho
        self.eta = loss.eta
        self.eps = torch.finfo(torch.float32).eps
        self.grad_clip = loss.grad_clip
        self.target_update_interval = loss.target_update_interval
        self.clip_auxiliary_loss_beta = loss.clip_auxiliary_loss_beta
        self.kl_beta = loss.kl_beta
        self.kl_balancing_mix = loss.kl_balancing_mix
        self.temperature = temperature
        self.bc_alpha = loss.bc_alpha
        self.actor_alpha = loss.actor_alpha

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gripper_control = gripper_control
        self.automatic_optimization = False

        # auxiliary losses
        self.use_clip_auxiliary_loss = use_clip_auxiliary_loss
        if use_clip_auxiliary_loss:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.proj_vis_lang = hydra.utils.instantiate(proj_vis_lang)

        self.use_bc_loss = use_bc_loss

        self.save_hyperparameters(ignore=["world_model"])

        # for inference
        self.rollout_step_counter = 0
        self.replan_freq = replan_freq
        self.latent_goal = None
        self.plan = None
        self.lang_embeddings = None

    @staticmethod
    def setup_input_sizes(
        world_model,
        plan_proposal,
        plan_recognition,
        visual_goal,
        action_decoder,
        critic,
        distribution,
    ):
        """
        Configure the input feature sizes of the respective parts of the network.

        Args:
            perceptual_encoder: DictConfig for perceptual encoder.
            plan_proposal: DictConfig for plan proposal network.
            plan_recognition: DictConfig for plan recognition network.
            visual_goal: DictConfig for visual goal encoder.
            action_decoder: DictConfig for action decoder network.
            distribution: DictConfig for plan distribution (continuous or discrete).
        """
        latent_size = world_model.decoder.in_dim

        plan_proposal.perceptual_features = 128  # latent_size
        plan_recognition.in_features = 128  # latent_size
        visual_goal.in_features = 128  # latent_size
        action_decoder.perceptual_features = latent_size
        critic.perceptual_features = latent_size

        if distribution.dist == "discrete":
            plan_proposal.plan_features = distribution.class_size * distribution.category_size
            plan_recognition.plan_features = distribution.class_size * distribution.category_size
            action_decoder.plan_features = distribution.class_size * distribution.category_size
            critic.plan_features = distribution.class_size * distribution.category_size
        elif distribution.dist == "continuous":
            plan_proposal.plan_features = distribution.plan_features
            plan_recognition.plan_features = distribution.plan_features
            action_decoder.plan_features = distribution.plan_features
            critic.plan_features = distribution.plan_features

    def configure_optimizers(self):

        self.actor_params = (
            list(self.action_decoder.parameters())
            + list(self.perceptual_encoder.parameters())
            + list(self.visual_goal.parameters())
            + list(self.language_goal.parameters())
            + list(self.plan_proposal.parameters())
            + list(self.plan_recognition.parameters())
        )
        if self.use_clip_auxiliary_loss:
            self.actor_params += list(self.proj_vis_lang.parameters())
            self.actor_params += [self.logit_scale]

        actor_optimizer = hydra.utils.instantiate(self.actor_optimizer, params=self.actor_params)
        critic_optimizer = hydra.utils.instantiate(self.critic_optimizer, params=self.critic.parameters())

        optimizers = [actor_optimizer, critic_optimizer]
        return optimizers

    def get_action_trajectory(self, c_state, latent_goal, sampled_plan, latents):
        val_buffer, target_buffer, rewards, actions, entropy = [], [], [], [], []
        for k in range(self.seq_len):
            a_hat, ent = self.action_decoder.get_action(c_state, latent_goal, sampled_plan)
            v_hat, v_t = self.get_value(c_state, latent_goal, sampled_plan)
            c_state = self.wm_step(c_state, a_hat)

            if k < self.seq_len - 1:
                reward = max_cos(c_state[..., : self.latent_half], latents[k + 1][..., : self.latent_half])
            else:
                reward = torch.zeros_like(rewards[-1]).to(self.device)

            actions.append(a_hat)
            entropy.append(ent)
            val_buffer.append(v_hat.squeeze())
            target_buffer.append(v_t.squeeze())
            rewards.append(reward)

        actions = torch.stack(actions)
        entropy = torch.mean(torch.stack(entropy))
        values = torch.stack(val_buffer)

        return actions, entropy, values, target_buffer, rewards

    def forward_train(self, latents: Tensor, codes: Tensor, latent_goal: Tensor, gt_acts: Tensor, robot_obs: Tensor):

        # ------------Plan Proposal------------ #
        pp_state = self.plan_proposal(codes[0], latent_goal)
        pp_dist = self.dist.get_dist(pp_state)

        # ------------Plan Recognition------------ #
        pr_state, seq_feat = self.plan_recognition(codes)
        pr_dist = self.dist.get_dist(pr_state)

        sampled_plan = pr_dist.rsample()  # sample from recognition net
        if self.dist.dist == "discrete":
            sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)

        c_state = latents[0]
        actions, entropy, values, target_buffer, rewards = self.get_action_trajectory(
            c_state, latent_goal, sampled_plan, latents
        )
        if self.gripper_control:
            actions = tcp_to_world_frame(actions, robot_obs.float())

        unnorm_returns = lambda_return(rewards, target_buffer, self.lambda_gae, self.gamma)
        losses = self.loss(actions, gt_acts, values, unnorm_returns, entropy, pp_state, pr_state)

        return losses, rewards, unnorm_returns, actions, pp_state, pr_state, seq_feat

    def forward_val(self, latents: Tensor, codes: Tensor, latent_goal: Tensor, robot_obs: Tensor):

        # ------------Plan Proposal------------ #
        pp_state = self.plan_proposal(codes[0], latent_goal)
        pp_dist = self.dist.get_dist(pp_state)

        # ------------ Policy network ------------ #
        c_state = latents[0]
        sampled_plan_pp = self.dist.sample_latent_plan(pp_dist)  # sample from proposal net
        actions, entropy, values, target_buffer, rewards = self.get_action_trajectory(
            c_state, latent_goal, sampled_plan_pp, latents
        )
        if self.gripper_control:
            actions = tcp_to_world_frame(actions, robot_obs.float())
        unnorm_returns = MC_return(rewards, target_buffer[-1], norm=False, gamma=self.gamma, eps=self.eps)

        # ------------Plan Recognition------------ #
        pr_state, seq_feat = self.plan_recognition(codes)
        pr_dist = self.dist.get_dist(pr_state)
        sampled_plan_pr = self.dist.sample_latent_plan(pr_dist)  # sample from recognition net

        loss_kl = self.kl_loss(pp_state, pr_state)
        losses = {"loss_kl": loss_kl}
        return losses, rewards, unnorm_returns, actions, pp_state, pr_state, seq_feat, sampled_plan_pp, sampled_plan_pr

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        critic_loss, actor_loss, entropy_loss, kl_loss, lang_clip_loss, bc_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        actor_optimizer, critic_optimizer = self.optimizers()

        if self.global_step % self.target_update_interval == 0:
            self.critic.update_critic_target()

        batch_size: Dict[str, int] = {}
        total_bs = 0

        for self.modality_scope, dataset_batch in batch.items():

            if torch.rand(1) > 0.5:
                latents = dataset_batch["feature"]
            else:
                latents = torch.cat([dataset_batch["zero_feature"][0:1], dataset_batch["feature"][1:]], dim=0)

            codes = self.perceptual_encoder(latents)

            if "lang" in self.modality_scope:
                latent_goal = self.language_goal(dataset_batch["lang"])
            else:
                latent_goal = self.visual_goal(codes[-1])

            losses, rewards, returns, ac_actions, pp_state, pr_state, seq_feat = self.forward_train(
                latents,
                codes,
                latent_goal,
                dataset_batch["actions"]["rel_actions"],
                dataset_batch["state_info"]["robot_obs"],
            )

            if "lang" in self.modality_scope:
                if not torch.any(dataset_batch["use_for_aux_lang_loss"]):
                    batch_size["aux_lang"] = 1
                else:
                    batch_size["aux_lang"] = torch.sum(dataset_batch["use_for_aux_lang_loss"]).detach()  # type:ignore

                if self.use_clip_auxiliary_loss:
                    lang_clip_loss = self.clip_auxiliary_loss(
                        seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"]
                    )

            critic_loss += losses["loss_critic"]
            actor_loss += losses["loss_actor"]
            entropy_loss += losses["loss_entropy"]
            kl_loss += losses["loss_kl"]
            bc_loss += losses["loss_bc"]
            batch_size[self.modality_scope] = dataset_batch["actions"]["rel_actions"].shape[1]
            total_bs += dataset_batch["actions"]["rel_actions"].shape[1]

            with torch.no_grad():
                act_mse = action_mse(dataset_batch["actions"]["rel_actions"][:-1], ac_actions[:-1])
                metrics = {
                    "metric_return-unnorm": returns.mean(),
                    "metric_reward-latent": torch.stack(rewards).mean(),
                    "metric_action-mse": sum(act_mse) / len(act_mse),
                }
                self.log(
                    f"train/critic_loss_{self.modality_scope}",
                    losses["loss_critic"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size[self.modality_scope],
                )
                self.log(
                    f"train/actor_loss_{self.modality_scope}",
                    losses["loss_actor"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size[self.modality_scope],
                )
                self.log(
                    f"train/entropy_loss_{self.modality_scope}",
                    losses["loss_entropy"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size[self.modality_scope],
                )
                self.log(
                    f"train/kl_loss_{self.modality_scope}",
                    losses["loss_kl"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size[self.modality_scope],
                )
                self.log(
                    f"train/bc_loss_{self.modality_scope}",
                    losses["loss_bc"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size[self.modality_scope],
                )
                self.log(
                    f"train/return-unnorm_{self.modality_scope}",
                    metrics["metric_return-unnorm"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size[self.modality_scope],
                )
                self.log(
                    f"train/reward-latent_{self.modality_scope}",
                    metrics["metric_reward-latent"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size[self.modality_scope],
                )
                self.log(
                    f"train/action_mse_{self.modality_scope}",
                    metrics["metric_action-mse"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size[self.modality_scope],
                )

        critic_loss = critic_loss / len(batch)
        actor_loss = actor_loss / len(batch)
        entropy_loss = entropy_loss / len(batch)
        kl_loss = kl_loss / len(batch)
        bc_loss = bc_loss / len(batch)
        loss_policy = actor_loss + bc_loss + kl_loss - entropy_loss

        if self.use_clip_auxiliary_loss:
            loss_policy = loss_policy + self.clip_auxiliary_loss_beta * lang_clip_loss
            self.log(
                "train/lang_clip_loss",
                self.clip_auxiliary_loss_beta * lang_clip_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size["aux_lang"],
                sync_dist=True,
            )

        critic_optimizer.zero_grad()
        self.manual_backward(critic_loss, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        critic_optimizer.step()

        actor_optimizer.zero_grad()
        self.manual_backward(loss_policy)
        torch.nn.utils.clip_grad_norm_(self.actor_params, self.grad_clip)
        actor_optimizer.step()

        self.log("train/critic_loss", critic_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        self.log("train/kl_loss", kl_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        self.log("train/actor_loss", actor_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        self.log("train/bc_loss", bc_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        self.log("train/entropy_loss", entropy_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        self.log("train/policy_loss", loss_policy, on_step=False, on_epoch=True, batch_size=total_bs)

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        batch_size: Dict[str, int] = {}
        total_bs = 0
        output = {}

        for self.modality_scope, dataset_batch in batch.items():
            latents = torch.cat([dataset_batch["zero_feature"][0:1], dataset_batch["feature"][1:]], dim=0)
            codes = self.perceptual_encoder(latents)
            if "lang" in self.modality_scope:
                latent_goal = self.language_goal(dataset_batch["lang"])
            else:
                latent_goal = self.visual_goal(codes[-1])

            losses, rewards, returns, ac_actions, pp_state, pr_state, seq_feat, sampled_plan_pp, sampled_plan_pr = (
                self.forward_val(
                    latents,
                    codes,
                    latent_goal,
                    dataset_batch["state_info"]["robot_obs"],
                )
            )

            batch_size[self.modality_scope] = dataset_batch["actions"]["rel_actions"].shape[1]
            total_bs += dataset_batch["actions"]["rel_actions"].shape[1]

            if "lang" in self.modality_scope and self.use_clip_auxiliary_loss:
                val_pred_clip_loss = self.clip_auxiliary_loss(
                    seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"]
                )
                self.log(
                    "val/lang_clip_loss",
                    val_pred_clip_loss,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size[self.modality_scope],
                )

            act_mse = action_mse(dataset_batch["actions"]["rel_actions"][:-1], ac_actions[:-1])
            metrics = {
                "metric_return-unnorm": returns.mean(),
                "metric_reward-latent": torch.stack(rewards[:-1]).mean(),
                "metric_action-mse": sum(act_mse) / len(act_mse),
            }
            self.log(
                f"val/kl_loss_{self.modality_scope}",
                losses["loss_kl"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
            self.log(
                f"val/return-unnorm_{self.modality_scope}",
                metrics["metric_return-unnorm"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
            self.log(
                f"val/reward-latent_{self.modality_scope}",
                metrics["metric_reward-latent"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
            self.log(
                f"val/action_mse_{self.modality_scope}",
                metrics["metric_action-mse"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
            output[f"sampled_plan_pp_{self.modality_scope}"] = sampled_plan_pp
            output[f"sampled_plan_pr_{self.modality_scope}"] = sampled_plan_pr
            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]
        return output

    def loss(self, acts, gt_acts, values, returns, entropy, pp_state, pr_state) -> Dict[str, Tensor]:

        loss_critic = self.critic_loss(values, returns)
        loss_actor = self.actor_loss(returns)
        loss_entropy = self.entropy_loss(entropy)
        loss_kl = self.kl_loss(pp_state, pr_state)

        if self.use_bc_loss:
            loss_bc = self.bc_loss(acts, gt_acts)
        else:
            loss_bc = torch.tensor(0.0).to(self.device)

        losses = {
            "loss_critic": loss_critic,
            "loss_actor": loss_actor,
            "loss_entropy": loss_entropy,
            "loss_kl": loss_kl,
            "loss_bc": loss_bc,
        }

        return losses

    def critic_loss(self, values, returns):
        loss_critic = 0.5 * F.mse_loss(values[:-1], returns.detach()[:-1])

        return loss_critic

    def actor_loss(self, returns):
        # dynamics backpropagation
        loss_actor = self.actor_alpha * (-returns[:-1]).mean()

        return loss_actor

    def bc_loss(self, acts, gt_acts):
        loss_bc = self.bc_alpha * F.mse_loss(acts[:-1], gt_acts[:-1])
        return loss_bc

    def entropy_loss(self, entropy: Tensor) -> Tensor:
        return entropy.mean() * self.eta

    def kl_loss(self, pp_state: State, pr_state: State) -> torch.Tensor:
        """
        Compute the KL divergence loss between the distributions of the plan recognition and plan proposal network.
        We use KL balancing similar to "MASTERING ATARI WITH DISCRETE WORLD MODELS" by Hafner et al.
        (https://arxiv.org/pdf/2010.02193.pdf)

        Args:
            pp_state: Namedtuple containing the parameters of the distribution produced by plan proposal network.
            pr_state: Namedtuple containing the parameters of the distribution produced by plan recognition network.

        Returns:
            Scaled KL loss.
        """
        pp_dist = self.dist.get_dist(pp_state)  # prior
        pr_dist = self.dist.get_dist(pr_state)  # posterior
        # @fixme: do this more elegantly
        kl_lhs = D.kl_divergence(self.dist.get_dist(self.dist.detach_state(pr_state)), pp_dist).mean()
        kl_rhs = D.kl_divergence(pr_dist, self.dist.get_dist(self.dist.detach_state(pp_state))).mean()

        alpha = self.kl_balancing_mix
        kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs
        kl_loss_scaled = kl_loss * self.kl_beta
        return kl_loss_scaled

    def clip_auxiliary_loss(self, seq_vis_feat, encoded_lang, use_for_aux_loss):
        """
        CLIP style contrastive loss, adapted from 'Learning transferable visual models from natural language
        supervision' by Radford et al.
        We maximize the cosine similarity between the visual features of the sequence i and the corresponding language
        features while, at the same time, minimizing the cosine similarity between the current visual features and other
        language instructions in the same batch.

        Args:
            seq_vis_feat: Visual embedding.
            encoded_lang: Language goal embedding.
            use_for_aux_loss: Mask of which sequences in the batch to consider for auxiliary loss.

        Returns:
            Contrastive loss.
        """
        assert self.use_clip_auxiliary_loss is not None
        skip_batch = False
        if use_for_aux_loss is not None:
            if not torch.any(use_for_aux_loss):
                # Hack for avoiding a crash when using ddp. Loss gets multiplied with 0 at the end of method to
                # effectively skip whole batch. We do a dummy forward pass, to prevent ddp from complaining.
                # see https://github.com/pytorch/pytorch/issues/43259
                skip_batch = True
                seq_vis_feat = seq_vis_feat[0:1]
                encoded_lang = encoded_lang[0:1]
            else:
                seq_vis_feat = seq_vis_feat[use_for_aux_loss]
                encoded_lang = encoded_lang[use_for_aux_loss]
        image_features, lang_features = self.proj_vis_lang(seq_vis_feat, encoded_lang)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # symmetric loss function
        labels = torch.arange(logits_per_image.shape[0], device=text_features.device)
        loss_i = cross_entropy(logits_per_image, labels)
        loss_t = cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        if skip_batch:
            loss *= 0
        return loss

    def get_value(self, x_c: Tensor, x_g: Tensor, x_p: Tensor) -> Tensor:
        x = torch.cat([x_c, x_g, x_p], dim=-1)
        return self.critic(x)

    def wm_step(self, latent, action):
        h, z = latent.split([self.latent_half, self.latent_half], -1)
        pp, (h, z) = self.wm.dream(action, (h, z), temperature=self.temperature)
        next_latent = torch.cat((h, z), dim=-1)
        return next_latent

    def reset(self):
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0
        self.pre_action = torch.zeros((1, 1, 7)).to(self.device)
        self.pre_action[:, :, -1] = 1.0
        self.wm_in_state = self.wm.rssm_core.init_state(1)
        self.wm_reset = torch.ones((1, 1, 1), dtype=torch.bool)

    @torch.no_grad()
    def step(self, obs, goal):

        if self.rollout_step_counter % self.replan_freq == 0:

            if isinstance(goal, str):
                embedded_lang = torch.from_numpy(self.lang_embeddings[goal]).to(self.device).squeeze(0).float()
                self.latent_goal = self.language_goal(embedded_lang)

            else:
                raise NotImplementedError("Only language goals are supported for now.")

            c_state, self.wm_in_state = self.obs_to_wm_latent(obs, self.wm_in_state, self.wm_reset, self.pre_action)
            self.plan = self.get_pp_plan(c_state, self.latent_goal)

        else:
            c_state, self.wm_in_state = self.obs_to_wm_latent(obs, self.wm_in_state, self.wm_reset, self.pre_action)

        local_action, _ = self.action_decoder.get_action(c_state, self.latent_goal, self.plan)

        if self.gripper_control:
            global_action = tcp_to_world_frame(local_action.unsqueeze(0), obs["robot_obs_raw"].unsqueeze(0).float())
            action = global_action
        else:
            action = local_action

        self.wm_reset = torch.zeros((1, 1, 1), dtype=torch.bool)
        self.pre_action = local_action.unsqueeze(0)
        self.rollout_step_counter += 1
        return action

    @torch.no_grad()
    def obs_to_wm_latent(self, obs, in_state, reset, action):
        action[:, :, -1] = 1 if action[:, :, -1] > 0 else -1
        features, out_state = self.wm.infer_features(
            obs["rgb_obs"]["rgb_static"],
            obs["rgb_obs"]["rgb_gripper"],
            obs["robot_obs"],
            action,
            obs["robot_obs_raw"],
            reset,
            in_state,
            local_act=action,
        )

        return features[0], out_state

    def get_pp_plan(self, latent_obs: dict, latent_goal: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use plan proposal network to sample new plan using a visual goal embedding.

        Args:
            obs: Observation from environment.
            goal: Embedded language instruction.

        Returns:
            sampled_plan: Sampled plan.
            latent_goal: Encoded language goal.
        """
        with torch.no_grad():
            code = self.perceptual_encoder(latent_obs)
            # ------------Plan Proposal------------ #
            pp_state = self.plan_proposal(code, latent_goal)
            pp_dist = self.dist.get_dist(pp_state)
            sampled_plan = self.dist.sample_latent_plan(pp_dist)
        return sampled_plan

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    def on_save_checkpoint(self, checkpoint):
        keys_to_remove = [key for key in checkpoint["state_dict"].keys() if key.startswith("clip.")]
        for key in keys_to_remove:
            del checkpoint["state_dict"][key]

    def load_lang_embeddings(self, embeddings_path):
        """
        This has to be called before inference. Loads the lang embeddings from the dataset.

        Args:
            embeddings_path: Path to <dataset>/validation/embeddings.npy
        """
        embeddings = np.load(embeddings_path, allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}
