import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticTSC
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCriticTSC

    def __init__(self,
                 actor_critic,
                 actor_critic_bbc,
                 estimator,
                 estimator_paras,
                 depth_encoder,
                 depth_encoder_paras,
                 depth_actor,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 dagger_update_freq=20,
                 priv_reg_coef_schedual=[0, 0, 0],
                 **kwargs
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.actor_critic_bbc = actor_critic_bbc
        self.actor_critic_bbc.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Adaptation
        self.hist_encoder_optimizer = optim.Adam(self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate)
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0

        # Estimator
        self.estimator = estimator
        self.priv_states_dim = estimator_paras["priv_states_dim"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_auxiliary = estimator_paras["num_auxiliary"]
        self.num_scan = estimator_paras["num_scan"]
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_paras["learning_rate"])
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]

        # Depth encoder
        self.if_depth = depth_encoder != None
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(),
                                                      lr=depth_encoder_paras["learning_rate"])
            self.depth_encoder_paras = depth_encoder_paras
            self.depth_actor = depth_actor
            self.depth_actor_optimizer = optim.Adam([*self.depth_actor.parameters(), *self.depth_encoder.parameters()],
                                                    lr=depth_encoder_paras["learning_rate"])
            self.byol_optimizer = torch.optim.Adam(self.depth_encoder.byol_learner.parameters(), lr=depth_encoder_paras["learning_rate_byol"])
        self.CE_loss = nn.CrossEntropyLoss().to(device)
        self.num_actions_d = self.actor_critic.num_actions_d

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
                                      action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, info, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values, use proprio to compute estimated priv_states then actions, but store true priv_states
        if self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])
            obs_est[:,
            self.num_prop + self.num_auxiliary + self.num_scan:self.num_prop + self.num_auxiliary + self.num_scan + self.priv_states_dim] = priv_states_estimated
            self.transition.actions = self.actor_critic.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()

        self.transition.actions_log_prob_d = self.actor_critic.get_actions_log_prob_d(
            self.transition.actions[:, 0]).detach()
        self.transition.actions_log_prob_c = self.actor_critic.get_actions_log_prob_c(
            self.transition.actions[:, 1:]).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        return self.transition.actions

    def act_bbc(self, obs):
        # Compute the actions and values, use proprio to compute estimated priv_states then actions, but store true priv_states
        if self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])
            obs_est[:,
            self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.priv_states_dim] = priv_states_estimated
            actions = self.actor_critic_bbc.act_inference(obs_est, hist_encoding=True).detach()
        else:
            actions = self.actor_critic_bbc.act_inference(obs, hist_encoding=True).detach()
        return actions

    def process_env_step(self, rewards, dones, infos):
        rewards_total = rewards.clone()

        self.transition.rewards = rewards_total.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

        return rewards_total

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimator_loss = 0
        mean_discriminator_loss = 0
        mean_discriminator_acc = 0
        mean_priv_reg_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
             old_actions_log_prob_d_batch, old_actions_log_prob_c_batch, old_mu_batch, old_sigma_batch,
             hid_states_batch, masks_batch) in generator:

            self.actor_critic.act(obs_batch.detach(), masks=masks_batch,
                                  hidden_states=hid_states_batch[0])  # match distribution dimension

            actions_log_prob_d_batch = self.actor_critic.get_actions_log_prob_d(actions_batch[:, 0])
            actions_log_prob_c_batch = self.actor_critic.get_actions_log_prob_c(actions_batch[:, 1:])
            value_batch = self.actor_critic.evaluate(critic_obs_batch.detach(), masks=masks_batch,
                                                     hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch_c = self.actor_critic.entropy_c
            entropy_batch_d = self.actor_critic.entropy_d
            entropy_batch = entropy_batch_c + entropy_batch_d

            # Adaptation module update
            priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
            with torch.inference_mode():
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
            priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
            priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]

            # Estimator
            priv_states_predicted = self.estimator(obs_batch[:, :self.num_prop])  # obs in batch is with true priv_states
            estimator_loss = (priv_states_predicted - obs_batch[:, self.num_prop + self.num_auxiliary + self.num_scan:self.num_prop + self.num_auxiliary + self.num_scan+self.priv_states_dim]).pow(2).mean()
            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                    torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                    2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio_d = torch.exp(actions_log_prob_d_batch - torch.squeeze(old_actions_log_prob_d_batch))
            surrogate_d = -torch.squeeze(advantages_batch) * ratio_d
            surrogate_d_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio_d, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss_d = torch.max(surrogate_d, surrogate_d_clipped).mean()

            ratio_c = torch.exp(actions_log_prob_c_batch - torch.squeeze(old_actions_log_prob_c_batch))
            surrogate_c = -torch.squeeze(advantages_batch) * ratio_c
            surrogate_c_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio_c, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss_c = torch.max(surrogate_c, surrogate_c_clipped).mean()

            surrogate_loss = surrogate_loss_d + surrogate_loss_c

            # Value function loss
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

            loss = surrogate_loss + \
                   self.value_loss_coef * value_loss - \
                   self.entropy_coef * entropy_batch.mean() + \
                   priv_reg_coef * priv_reg_loss + \
                   0.0 * b_loss.mean()
            # loss = self.teacher_alpha * imitation_loss + (1 - self.teacher_alpha) * loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimator_loss += estimator_loss.item()
            mean_priv_reg_loss += priv_reg_loss.item()
            mean_discriminator_loss += 0
            mean_discriminator_acc += 0

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_discriminator_loss /= num_updates
        mean_discriminator_acc /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, priv_reg_coef

    def update_dagger(self):
        mean_hist_latent_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
             old_actions_log_prob_d_batch, old_actions_log_prob_c_batch, old_mu_batch, old_sigma_batch,
             hid_states_batch, masks_batch) in generator:
            # TODO: self.actor_critic.act步骤是否必要？
            with torch.inference_mode():
                self.actor_critic.act(obs_batch, hist_encoding=True, masks=masks_batch,
                                      hidden_states=hid_states_batch[0])

            # Adaptation module update
            with torch.inference_mode():
                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
            hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()

            mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def update_depth_encoder(self, depth_latent_batch, scandots_latent_batch):
        # Depth encoder ditillation
        if self.if_depth:
            # TODO: needs to save hidden states
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()

            self.depth_encoder_optimizer.zero_grad()
            depth_encoder_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), self.max_grad_norm)
            self.depth_encoder_optimizer.step()
            return depth_encoder_loss.item()

    def update_depth_actor(self, actions_student_batch, actions_teacher_batch, yaw_student_batch, yaw_teacher_batch,
                           obst_type_buffer_student, obst_type_buffer_teacher, depth_batch):
        if self.if_depth:
            depth_actor_d_loss = self.CE_loss(actions_student_batch[:, :self.num_actions_d], actions_teacher_batch[:, 0].detach().to(torch.int64))
            depth_actor_c_loss = (actions_teacher_batch[:, 1:].detach() - actions_student_batch[:, self.num_actions_d:]).norm(p=2, dim=1).mean()
            depth_actor_loss = depth_actor_d_loss + depth_actor_c_loss
            yaw_scale_factors = torch.tensor([2.0, 0.5], device=yaw_teacher_batch.device)
            yaw_loss = ((yaw_teacher_batch.detach() - yaw_student_batch) * yaw_scale_factors).norm(p=2, dim=1).mean()
            obst_type_loss = self.CE_loss(obst_type_buffer_student, torch.argmax(obst_type_buffer_teacher, dim=-1))

            loss = depth_actor_loss + yaw_loss + obst_type_loss

            self.depth_actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.depth_actor.parameters(), self.max_grad_norm)
            self.depth_actor_optimizer.step()

            num_samples = depth_batch.size(0)
            batch_size = num_samples // 6  # 4608 // 6
            random_indices = torch.randperm(num_samples)
            depth_batch = depth_batch[random_indices]
            mean_byol_loss = 0
            for i in range(0, num_samples, batch_size):
                depth_batch_minibatch = depth_batch[i:i + batch_size]
                byol_loss = self.depth_encoder.byol_learner(depth_batch_minibatch)
                self.byol_optimizer.zero_grad()
                byol_loss.backward()
                self.byol_optimizer.step()
                mean_byol_loss += byol_loss.item()
                self.depth_encoder.byol_learner.update_moving_average()
            mean_byol_loss = mean_byol_loss / (num_samples // batch_size)
            return depth_actor_loss.item(), yaw_loss.item(), obst_type_loss.item(), mean_byol_loss

    def update_depth_both(self, depth_latent_batch, scandots_latent_batch, actions_student_batch,
                          actions_teacher_batch):
        if self.if_depth:
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()

            depth_loss = depth_encoder_loss + depth_actor_loss

            self.depth_actor_optimizer.zero_grad()
            depth_loss.backward()
            nn.utils.clip_grad_norm_([*self.depth_actor.parameters(), *self.depth_encoder.parameters()],
                                     self.max_grad_norm)
            self.depth_actor_optimizer.step()
            return depth_encoder_loss.item(), depth_actor_loss.item()

    def update_counter(self):
        self.counter += 1

    def compute_apt_reward(self, source, target):

        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        # sim_matrix = torch.norm(source[:, None, ::2].view(b1, 1, -1) - target[None, :, ::2].view(1, b2, -1), dim=-1, p=2)
        # sim_matrix = torch.norm(source[:, None, :2].view(b1, 1, -1) - target[None, :, :2].view(1, b2, -1), dim=-1, p=2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)

        reward, _ = sim_matrix.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)  # (b1, )
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1)  # (b1,)
        reward = torch.log(reward + 1.0)
        return reward
