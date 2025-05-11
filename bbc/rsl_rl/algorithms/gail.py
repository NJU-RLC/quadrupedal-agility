import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer


class SSInfoGAIL:
    actor_critic: ActorCritic

    def __init__(self,
                 env,
                 actor_critic,
                 discriminator,
                 estimator,
                 estimator_paras,
                 motion_loader,
                 disc_normalizer,
                 disc_history_len,
                 disc_obs_len,
                 num_disc_obs,
                 obs_disc_weight_step,
                 disc_loss_function=None,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 surrogate_loss_coef=1.,
                 value_loss_coef=5.,
                 entropy_coef=0.,
                 bounds_loss_coef=10.,
                 disc_coef=5.,
                 disc_logit_reg=0.05,
                 disc_grad_penalty=0.2,
                 disc_weight_decay=0.0001,
                 lr_ac=1e-3,
                 lr_disc=1e-3,
                 lr_q=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=False,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 disc_replay_buffer_size=100000,
                 min_std=None,
                 us_coef=1.0,
                 ss_coef=4.0,
                 prior_soft_coef=1e-3,
                 info_max_coef=2.0,
                 begin_rim=100,
                 priv_reg_coef_schedual=[0, 0.1, 0, 1],
                 priv_reg_coef_schedual_resume=[0, 0.1, 0, 1]
                 ):

        self.device = device
        self.env = env
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.lr_ac = lr_ac
        self.lr_disc = lr_disc
        self.lr_q = lr_q
        self.min_std = min_std
        self.dim_c = self.env.dim_c
        self.disc_loss_function = disc_loss_function
        self.disc_history_len = disc_history_len
        self.disc_obs_len = disc_obs_len
        self.num_disc_obs = num_disc_obs
        self.obs_disc_weight_step = obs_disc_weight_step

        # Discriminator components
        self.disc = discriminator
        self.disc.to(self.device)
        self.disc_storage = ReplayBuffer(self.env.num_obs_disc, self.dim_c, disc_obs_len,
                                         disc_replay_buffer_size, device)
        self.motion_loader = motion_loader
        self.disc_normalizer = disc_normalizer

        self.num_prop = self.env.cfg.env.num_prop
        self.num_explicit = self.env.cfg.env.num_explicit
        self.num_latent = self.env.cfg.env.num_latent
        self.num_hist = self.env.cfg.env.history_len
        self.num_command = self.env.cfg.env.num_command

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.transition = None
        self.storage = None

        # Optimizer for policy and discriminator.
        params_ac = [{'params': self.actor_critic.parameters(), 'name': 'actor_critic'}]
        self.optim_ac = optim.Adam(params_ac, lr=self.lr_ac)
        # Adaptation
        self.optim_hist_encoder = optim.Adam(self.actor_critic.history_encoder.parameters(),
                                             lr=estimator_paras["learning_rate"])
        # Estimator
        self.estimator = estimator
        self.optim_estimator = optim.Adam(self.estimator.parameters(), lr=estimator_paras["learning_rate"])

        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.priv_reg_counter = 0
        self.train_with_estimated_explicit = estimator_paras["train_with_estimated_explicit"]

        params_d = [
            {'params': self.disc.trunk.parameters(),
             'weight_decay': 1e-3, 'momentum': 0.9, 'name': 'trunk'},
            {'params': self.disc.linear.parameters(),
             'weight_decay': 1e-3, 'momentum': 0.9, 'name': 'head'}
        ]
        params_q_eps = [
            {'params': self.disc.trunk.parameters(),
             'weight_decay': 1e-3, 'momentum': 0.9, 'name': 'trunk'},
            {'params': self.disc.encoder_eps.parameters(),
             'weight_decay': 1e-3, 'momentum': 0.9, 'name': 'encoder_eps'},
        ]
        params_q_c = [
            {'params': self.disc.trunk.parameters(),
             'weight_decay': 1e-3, 'momentum': 0.9, 'name': 'trunk'},
            {'params': self.disc.classifier.parameters(),
             'weight_decay': 1e-3, 'momentum': 0.9, 'name': 'classifier'}
        ]
        if self.disc_loss_function == "WassersteinLoss":
            self.optim_d = optim.RMSprop(params_d, lr=self.lr_disc)
        else:
            self.optim_d = optim.Adam(params_d, lr=self.lr_disc)
        self.optim_q_eps = optim.Adam(params_q_eps, lr=self.lr_q)
        self.optim_q_c = optim.Adam(params_q_c, lr=self.lr_q)

        # GAIL parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.surrogate_loss_coef = surrogate_loss_coef
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.bounds_loss_coef = bounds_loss_coef
        self.disc_coef = disc_coef
        self.disc_logit_reg = disc_logit_reg
        self.disc_grad_penalty = disc_grad_penalty
        self.disc_weight_decay = disc_weight_decay
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.us_coef = us_coef
        self.ss_coef = ss_coef
        self.prior_soft_coef = prior_soft_coef
        self.info_max_coef_on = 0
        self.info_max_coef = info_max_coef
        self.learning_steps = 0
        self.begin_rim = begin_rim

        # Loss functions
        self.CE_loss = nn.CrossEntropyLoss().to(device)
        self.MSELoss = nn.MSELoss().to(device)
        self.L1Loss = nn.L1Loss().to(device)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.transition = RolloutStorage.Transition(num_envs, actor_obs_shape, critic_obs_shape,
                                                    action_shape, self.device)
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
                                      action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, hist_encoding=False):
        if self.actor_critic.is_recurrent:  # False
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()

        if self.train_with_estimated_explicit:
            obs_est = aug_obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])
            obs_est[:, self.num_prop:self.num_prop + self.num_explicit] = priv_states_estimated
            self.transition.actions = self.actor_critic.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.actor_critic.act(aug_obs, hist_encoding).detach()

        self.transition.values = self.actor_critic.evaluate(aug_critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, obs_disc_history_buf):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        self.disc_storage.insert(
            obs_disc_history_buf.view(obs_disc_history_buf.shape[0], -1), self.env.latent_eps, self.env.latent_c)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        aug_last_critic_obs = last_critic_obs.detach()
        last_values = self.actor_critic.evaluate(aug_last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    @staticmethod
    def weighted_bce_loss(predictions, targets, weights):
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        weighted_bce = torch.mean(weights * bce_loss)
        return weighted_bce

    @staticmethod
    def weighted_mse_loss(predictions, targets, weights):
        mse_loss = F.mse_loss(predictions, targets, reduction='none')
        weighted_mse = torch.mean(weights * mse_loss)
        return weighted_mse

    def update(self):
        mean_surrogate_loss = 0
        mean_value_loss = 0
        mean_b_loss = 0
        mean_entropy_batch = 0
        mean_priv_reg_loss = 0
        mean_estimator_loss = 0
        mean_ss_loss = 0
        mean_info_max_loss = 0
        mean_disc_loss = 0
        mean_us_loss = 0
        mean_grad_pen_loss = 0
        mean_disc_logit_loss = 0
        mean_disc_weight_decay = 0
        mean_acc_lb = 0
        mean_acc_pi = 0
        mean_acc_exp = 0
        mean_acc_ulb = 0
        self.learning_steps += 1

        if self.learning_steps >= self.begin_rim:
            self.info_max_coef_on = min(self.info_max_coef * (self.learning_steps - self.begin_rim) * 1 /
                                        10000, self.info_max_coef)

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        discriminator_num_mini_batches = self.num_learning_epochs * self.num_mini_batches * 4
        disc_policy_generator = self.disc_storage.feed_forward_generator(
            discriminator_num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
            discriminator_num_mini_batches)
        disc_expert_generator_lb = self.motion_loader.feed_forward_generator_lb(
            discriminator_num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
            discriminator_num_mini_batches)
        disc_expert_generator_ulb = self.motion_loader.feed_forward_generator_ulb(
            discriminator_num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
            discriminator_num_mini_batches)

        for sample in generator:
            surrogate_loss, value_loss, b_loss, entropy_batch, priv_reg_loss, estimator_loss = (
                self.update_actor_critic(sample))
            mean_surrogate_loss += surrogate_loss.item()
            mean_value_loss += value_loss.item()
            mean_b_loss += b_loss.mean().item()
            mean_entropy_batch += entropy_batch.mean().item()
            mean_priv_reg_loss += priv_reg_loss.mean().item()
            mean_estimator_loss += estimator_loss.mean().item()

        for sample_disc_policy, sample_disc_expert_lb, sample_disc_expert_ulb in (
                zip(disc_policy_generator, disc_expert_generator_lb, disc_expert_generator_ulb)):
            (ss_loss, info_max_loss, disc_loss, us_loss, grad_pen_loss, disc_logit_loss, disc_weight_decay,
             acc_lb, acc_pi, acc_exp, acc_ulb) = self.update_ss_info_gail(sample_disc_policy, sample_disc_expert_lb,
                                                                          sample_disc_expert_ulb)

            mean_ss_loss += ss_loss.item()
            mean_info_max_loss += info_max_loss.item()
            mean_disc_loss += disc_loss.item()
            mean_us_loss += us_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_disc_logit_loss += disc_logit_loss.item()
            mean_disc_weight_decay += disc_weight_decay.item()
            mean_acc_lb += acc_lb.item()
            mean_acc_pi += acc_pi.item()
            mean_acc_exp += acc_exp.item()
            mean_acc_ulb += acc_ulb.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_surrogate_loss /= num_updates
        mean_value_loss /= num_updates
        mean_b_loss /= num_updates
        mean_entropy_batch /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_ss_loss /= discriminator_num_mini_batches
        mean_info_max_loss /= discriminator_num_mini_batches
        mean_disc_loss /= discriminator_num_mini_batches
        mean_us_loss /= discriminator_num_mini_batches
        mean_grad_pen_loss /= discriminator_num_mini_batches
        mean_disc_logit_loss /= discriminator_num_mini_batches
        mean_disc_weight_decay /= discriminator_num_mini_batches
        mean_acc_lb /= discriminator_num_mini_batches
        mean_acc_pi /= discriminator_num_mini_batches
        mean_acc_exp /= discriminator_num_mini_batches
        mean_acc_ulb /= discriminator_num_mini_batches

        self.storage.clear()
        self.priv_reg_counter += 1
        return (mean_surrogate_loss, mean_value_loss, mean_b_loss, mean_entropy_batch, mean_priv_reg_loss,
                mean_estimator_loss, mean_ss_loss, mean_info_max_loss, mean_disc_loss,
                mean_us_loss, mean_grad_pen_loss, mean_disc_logit_loss, mean_disc_weight_decay, mean_acc_lb,
                mean_acc_pi, mean_acc_exp, mean_acc_ulb)

    def update_actor_critic(self, sample):
        (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
         old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch) = sample

        aug_obs_batch = obs_batch.detach()
        self.actor_critic.act(aug_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])  # sample actions
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        aug_critic_obs_batch = critic_obs_batch.detach()
        value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, masks=masks_batch,
                                                 hidden_states=hid_states_batch[1])
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        entropy_batch = self.actor_critic.entropy

        obs_prop_batch = obs_batch[:, :self.num_prop]
        obs_explicit_batch = obs_batch[:, self.num_prop:self.num_prop + self.num_explicit]
        obs_latent_batch = obs_batch[:,
                           self.num_prop + self.num_explicit:self.num_prop + self.num_explicit + self.num_latent]
        obs_hist_batch = obs_batch[:,
                         self.num_prop + self.num_explicit + self.num_latent:self.num_prop + self.num_explicit + self.num_latent + self.num_hist * self.num_prop]

        # Adaptation module update
        priv_latent_batch = self.actor_critic.infer_priv_latent(obs_latent_batch)
        with torch.inference_mode():
            hist_latent_batch = self.actor_critic.infer_hist_latent(obs_hist_batch)
        priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
        priv_reg_stage = min(
            max((self.priv_reg_counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
        priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + \
                        self.priv_reg_coef_schedual[0]

        # Estimator
        priv_states_predicted = self.estimator(obs_prop_batch)  # obs in batch is with true priv_states
        estimator_loss = (priv_states_predicted - obs_explicit_batch).pow(2).mean()
        self.optim_estimator.zero_grad()
        estimator_loss.backward()
        nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
        self.optim_estimator.step()

        if self.desired_kl is not None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.e-5) +
                               (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) /
                               (2.0 * torch.square(sigma_batch)) - 0.5, dim=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.lr_ac = max(1e-5, self.lr_ac / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.lr_ac = min(1e-2, self.lr_ac * 1.5)
                for param_group in self.optim_ac.param_groups:
                    param_group['lr'] = self.lr_ac

        # Surrogate loss (actor)
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                           1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss (critic)
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        # bound loss
        soft_bound = 1.0
        mu_loss_high = torch.maximum(mu_batch - soft_bound,
                                     torch.tensor(0, device=self.device)) ** 2
        mu_loss_low = torch.minimum(mu_batch + soft_bound, torch.tensor(0, device=self.device)) ** 2
        b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)

        ppo_loss = (self.surrogate_loss_coef * surrogate_loss + self.value_loss_coef * value_loss +
                    self.bounds_loss_coef * b_loss.mean() - self.entropy_coef * entropy_batch.mean() +
                    priv_reg_coef * priv_reg_loss)

        self.optim_ac.zero_grad()
        ppo_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optim_ac.step()
        return surrogate_loss, value_loss, b_loss, entropy_batch, priv_reg_loss, estimator_loss

    def update_ss_info_gail(self, sample_disc_policy, sample_disc_expert_lb, sample_disc_expert_ulb):
        # Discriminator loss
        policy_state, policy_latent_eps, policy_latent_c = sample_disc_policy
        expert_state_lb, label_exp_lb = sample_disc_expert_lb
        expert_state_ulb = sample_disc_expert_ulb

        policy_state = policy_state.view(len(policy_state), self.disc_obs_len, -1).clone()
        expert_state_lb = expert_state_lb.view(len(expert_state_lb), self.disc_obs_len, -1).clone()
        expert_state_ulb = expert_state_ulb.view(len(expert_state_ulb), self.disc_obs_len, -1).clone()

        if self.env.task_obs_weight_decay:
            # lin_vel, ang_vel, key_pos, foot_contact
            policy_state[:, :, 3:9] *= self.env.task_obs_weight
            policy_state[:, :, 33:] *= self.env.task_obs_weight
            expert_state_lb[:, :, 3:9] *= self.env.task_obs_weight
            expert_state_lb[:, :, 33:] *= self.env.task_obs_weight
            expert_state_ulb[:, :, 3:9] *= self.env.task_obs_weight
            expert_state_ulb[:, :, 33:] *= self.env.task_obs_weight

        policy_state = policy_state[:, -self.disc_obs_len:, :].view(len(policy_state), -1)
        expert_state_lb = expert_state_lb[:, -self.disc_obs_len:, :].view(len(expert_state_lb), -1)
        expert_state_ulb = expert_state_ulb[:, -self.disc_obs_len:, :].view(len(expert_state_ulb), -1)

        multipliers = (torch.arange(self.disc_obs_len, dtype=torch.float32, device=self.device) *
                       self.obs_disc_weight_step + 1)
        multipliers = multipliers.view(1, -1, 1).repeat(len(policy_state), 1,
                                                        self.num_disc_obs).view(len(policy_state), -1)
        policy_state = policy_state * multipliers
        expert_state_lb = expert_state_lb * multipliers
        expert_state_ulb = expert_state_ulb * multipliers

        if self.disc_normalizer is not None:
            with torch.no_grad():
                policy_state = self.disc_normalizer.normalize_torch(policy_state, self.device)
                expert_state_lb = self.disc_normalizer.normalize_torch(expert_state_lb, self.device)
                expert_state_ulb = self.disc_normalizer.normalize_torch(expert_state_ulb, self.device)

        # Update semi-supervised encoder
        _, _, pred_c_lb = self.disc(expert_state_lb)
        ss_loss = self.CE_loss(pred_c_lb, label_exp_lb)

        # Update discriminator and unsupervised encoder
        policy_latent_c = F.one_hot(torch.argmax(policy_latent_c, dim=-1), num_classes=self.dim_c)
        logits_pi, eps, pred_c = self.disc(policy_state)
        logits_exp, _, pred_c_ulb = self.disc(expert_state_ulb)

        # estimate the prior
        pred_c_ulb_mean = torch.mean(pred_c_ulb, dim=0)
        self.env.prior_parameters = pred_c_ulb_mean.detach() * self.prior_soft_coef + \
                                    self.env.prior_parameters * (1 - self.prior_soft_coef)
        # normalized information maximization
        info_max_loss = torch.mean(-torch.sum(pred_c_ulb * torch.log(pred_c_ulb + 1e-20), dim=-1))

        weight_pi = torch.ones_like(eps)
        weight_exp = torch.ones_like(eps)

        if self.disc_loss_function == "BCEWithLogitsLoss":
            disc_exp_loss = self.weighted_bce_loss(logits_exp, torch.ones_like(logits_exp), weight_exp)
            disc_pi_loss = self.weighted_bce_loss(logits_pi, torch.zeros_like(logits_pi), weight_pi)
        elif self.disc_loss_function == "MSELoss":
            disc_exp_loss = self.weighted_mse_loss(logits_exp, torch.ones_like(logits_exp), weight_exp)
            disc_pi_loss = self.weighted_mse_loss(logits_pi, -1 * torch.ones_like(logits_pi), weight_pi)
        elif self.disc_loss_function == "WassersteinLoss":
            disc_exp_loss = -(weight_exp * logits_exp).mean()
            disc_pi_loss = (weight_pi * logits_pi).mean()
        else:
            raise ValueError("Unexpected loss function specified")

        disc_loss = 0.5 * (disc_pi_loss + disc_exp_loss)

        us_loss = self.L1Loss(eps, policy_latent_eps)

        # logit reg
        logit_weights = self.disc.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))

        # grad penalty
        sample_expert = expert_state_ulb.clone()
        sample_expert.requires_grad = True
        disc = self.disc.linear(self.disc.trunk(sample_expert))
        ones = torch.ones(disc.size(), device=disc.device)
        disc_demo_grad = torch.autograd.grad(disc, sample_expert,
                                             grad_outputs=ones,
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        grad_pen_loss = torch.mean(disc_demo_grad)

        # weight decay
        disc_weights = self.disc.get_disc_weights()
        disc_weights = torch.cat(disc_weights, dim=-1)
        disc_weight_decay = torch.sum(torch.square(disc_weights))

        # Compute total loss.
        loss = (self.ss_coef * ss_loss + self.info_max_coef_on * info_max_loss +
                self.disc_coef * disc_loss + self.us_coef * us_loss + self.disc_grad_penalty * grad_pen_loss +
                self.disc_logit_reg * disc_logit_loss + self.disc_weight_decay * disc_weight_decay)

        # Gradient step
        self.optim_d.zero_grad()
        self.optim_q_eps.zero_grad()
        self.optim_q_c.zero_grad()
        loss.backward()
        self.optim_d.step()
        self.optim_q_eps.step()
        self.optim_q_c.step()

        if not self.actor_critic.fixed_std and self.min_std is not None:  # True
            self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

        # calculate the running mean and std of the data stream
        if self.disc_normalizer is not None:
            self.disc_normalizer.update(policy_state.cpu().numpy())
            self.disc_normalizer.update(expert_state_lb.cpu().numpy())
            self.disc_normalizer.update(expert_state_ulb.cpu().numpy())

        with torch.no_grad():
            pred_c_lb = torch.argmax(pred_c_lb, dim=-1)
            acc_lb = torch.mean((pred_c_lb == label_exp_lb).float())
            acc_pi = (logits_pi < 0).float().mean()
            acc_exp = (logits_exp > 0).float().mean()
            pred_c = torch.argmax(pred_c, dim=-1)
            policy_latent_c = torch.argmax(policy_latent_c, dim=-1)
            acc_ulb = torch.mean((pred_c == policy_latent_c).float())

        return (ss_loss, info_max_loss, disc_loss, us_loss, grad_pen_loss, disc_logit_loss, disc_weight_decay,
                acc_lb, acc_pi, acc_exp, acc_ulb)

    def update_dagger(self):
        mean_hist_latent_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
            with torch.inference_mode():
                self.actor_critic.act(obs_batch, hist_encoding=True, masks=masks_batch,
                                      hidden_states=hid_states_batch[0])

            obs_latent_batch = obs_batch[:,
                               self.num_prop + self.num_explicit:self.num_prop + self.num_explicit + self.num_latent]
            obs_hist_batch = obs_batch[:,
                             self.num_prop + self.num_explicit + self.num_latent:self.num_prop + self.num_explicit + self.num_latent + self.num_hist * self.num_prop]

            # Adaptation module update
            with torch.inference_mode():
                priv_latent_batch = self.actor_critic.infer_priv_latent(obs_latent_batch)
            hist_latent_batch = self.actor_critic.infer_hist_latent(obs_hist_batch)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            self.optim_hist_encoder.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.history_encoder.parameters(), self.max_grad_norm)
            self.optim_hist_encoder.step()

            mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.priv_reg_counter += 1
        return mean_hist_latent_loss
