import os
import time
import statistics
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from rsl_rl.algorithms import SSInfoGAIL
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, Estimator
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.discriminator import Discriminator
from rsl_rl.datasets.motion_loader import MotionLoader
from rsl_rl.utils.utils import Normalizer
from tqdm import trange
from legged_gym.envs.base.legged_robot import compute_flat_key_pos


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.device = device
        self.env = env
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.disc_loss_function = self.alg_cfg["disc_loss_function"]
        self.reward_i_coef = train_cfg['runner']['reward_i_coef']
        self.reward_us_coef = train_cfg['runner']['reward_us_coef']
        self.reward_ss_coef = train_cfg['runner']['reward_ss_coef']
        self.reward_t_coef = train_cfg['runner']['reward_t_coef']
        self.disc_history_len = self.env.cfg.env.disc_history_len
        self.disc_obs_len = self.env.cfg.env.disc_obs_len
        self.obs_disc_weight_step = self.env.cfg.env.obs_disc_weight_step

        actor_critic_class = eval(self.cfg["policy_class_name"])

        num_prop = self.env.cfg.env.num_prop
        num_hist = self.env.cfg.env.history_len
        num_explicit = self.env.cfg.env.num_explicit
        num_latent = self.env.cfg.env.num_latent
        num_command = self.env.cfg.env.num_command

        num_actor_obs = self.env.num_obs
        num_critic_obs = self.env.num_obs + num_hist * num_prop
        num_disc_obs = self.env.num_obs_disc

        actor_critic: ActorCritic = actor_critic_class(num_actor_obs,
                                                       num_critic_obs,
                                                       self.env.num_actions,
                                                       num_prop, num_hist, num_explicit, num_latent, num_command,
                                                       **self.policy_cfg).to(self.device)

        estimator = Estimator(input_dim=num_prop, output_dim=num_explicit,
                              hidden_dims=self.estimator_cfg["hidden_dims"]).to(self.device)

        motion_loader = MotionLoader(device, time_between_frames=self.env.dt,
                                     motion_files_lb=self.cfg["motion_files_lb"],
                                     motion_files_ulb=self.cfg["motion_files_ulb"],
                                     mocap_category=self.env.mocap_category,
                                     num_preload_transitions=self.cfg["num_preload_transitions"],
                                     compute_flat_key_pos=compute_flat_key_pos,
                                     default_dof_pos=self.env.default_dof_pos, obs_scales=self.env.obs_scales,
                                     num_disc_obs=num_disc_obs, disc_obs_len=self.disc_obs_len,
                                     obs_disc_weight_step=self.obs_disc_weight_step,
                                     frame_duration_scale=self.env.cfg.env.frame_duration_scale)
        disc_normalizer = Normalizer(num_disc_obs * self.disc_obs_len)
        reward_i_normalizer = None
        if self.disc_loss_function == "WassersteinLoss":
            reward_i_normalizer = Normalizer(1)
        discriminator = Discriminator(
            self.env,
            num_disc_obs * self.disc_obs_len,
            num_disc_obs,
            len(self.env.mocap_category),
            self.env.dt,
            self.disc_loss_function,
            reward_i_normalizer,
            self.reward_i_coef,
            self.reward_us_coef,
            self.reward_ss_coef,
            self.reward_t_coef,
            self.disc_history_len,
            self.disc_obs_len,
            self.obs_disc_weight_step,
            train_cfg['runner']['disc_hidden_units'], device).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])
        min_std = (
                torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
                (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))

        self.alg: SSInfoGAIL = alg_class(self.env, actor_critic, discriminator, estimator,
                                         self.estimator_cfg, motion_loader, disc_normalizer,
                                         self.disc_history_len, self.disc_obs_len, num_disc_obs,
                                         self.obs_disc_weight_step, device=self.device, min_std=min_std, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.cfg["dagger_update_freq"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs + num_hist * num_prop],
                              [self.env.num_privileged_obs + num_hist * num_prop],
                              [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        disc_obs = self.env.get_disc_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, disc_obs = obs.to(self.device), critic_obs.to(self.device), disc_obs.to(self.device)
        obs_disc_history_buf = torch.stack([disc_obs] * self.disc_obs_len, dim=1)
        self.alg.actor_critic.train()
        self.alg.disc.train()

        ep_infos = []
        rew_buffer = deque(maxlen=100)
        rew_buffer_i = deque(maxlen=100)
        rew_buffer_us = deque(maxlen=100)
        rew_buffer_ss = deque(maxlen=100)
        rew_buffer_t = deque(maxlen=100)
        len_buffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_i = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_us = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_ss = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_t = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        pbar = trange(self.current_learning_iteration, tot_iter)
        for it in pbar:
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, hist_encoding)
                    next_obs, next_privileged_obs, rewards, dones, infos, reset_env_ids, terminal_disc_states = self.env.step(
                        actions)
                    next_disc_obs = self.env.get_disc_observations()
                    next_critic_obs = next_privileged_obs if next_privileged_obs is not None else next_obs
                    next_obs, next_critic_obs, next_disc_obs, rewards, dones = next_obs.to(
                        self.device), next_critic_obs.to(
                        self.device), next_disc_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    next_disc_obs_with_term = torch.clone(next_disc_obs)
                    next_disc_obs_with_term[reset_env_ids] = terminal_disc_states
                    obs_disc_history_buf = torch.cat([obs_disc_history_buf[:, 1:],
                                                      next_disc_obs_with_term.unsqueeze(1)], dim=1)

                    rewards, reward_i, reward_us, reward_ss, reward_t = self.alg.disc.predict_disc_reward(
                        rewards.unsqueeze(1), obs, obs_disc_history_buf, normalizer=self.alg.disc_normalizer)

                    self.alg.process_env_step(rewards, dones, infos, obs_disc_history_buf)

                    obs = torch.clone(next_obs)
                    critic_obs = torch.clone(next_critic_obs)
                    disc_obs = torch.clone(next_disc_obs)

                    obs_disc_history_buf[reset_env_ids] = torch.stack([disc_obs[reset_env_ids]] *
                                                                      self.disc_obs_len, dim=1)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        # cur_reward_sum += (reward_i + reward_us + reward_ss + reward_t)
                        cur_reward_sum += rewards
                        cur_reward_i += reward_i
                        cur_reward_us += reward_us
                        cur_reward_ss += reward_ss
                        cur_reward_t += reward_t
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rew_buffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_buffer_i.extend(cur_reward_i[new_ids][:, 0].cpu().numpy().tolist())
                        rew_buffer_us.extend(cur_reward_us[new_ids][:, 0].cpu().numpy().tolist())
                        rew_buffer_ss.extend(cur_reward_ss[new_ids][:, 0].cpu().numpy().tolist())
                        rew_buffer_t.extend(cur_reward_t[new_ids][:, 0].cpu().numpy().tolist())
                        len_buffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_reward_i[new_ids] = 0
                        cur_reward_us[new_ids] = 0
                        cur_reward_ss[new_ids] = 0
                        cur_reward_t[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            (mean_surrogate_loss, mean_value_loss, mean_b_loss, mean_entropy_batch, mean_priv_reg_loss,
             mean_estimator_loss, mean_ss_loss, mean_info_max_loss, mean_disc_loss,
             mean_us_loss, mean_grad_pen_loss, mean_disc_logit_loss, mean_disc_weight_decay, mean_acc_lb,
             mean_acc_pi, mean_acc_exp, mean_acc_ulb) = self.alg.update()

            if hist_encoding:
                mean_hist_latent_loss = self.alg.update_dagger()

            if self.env.task_obs_weight_decay_steps:
                self.env.task_obs_weight = max(0, self.env.task_obs_weight - 1.0 / self.env.task_obs_weight_decay_steps)

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals(), pbar)
            if (it + 1) % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model.pt'))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model.pt'))

    def log(self, locs, pbar):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor) / self.env.reward_scales[key[4:]]
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/surrogate_loss', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/value_loss', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/b_loss', locs['mean_b_loss'], locs['it'])
        self.writer.add_scalar('Loss/entropy_batch', locs['mean_entropy_batch'], locs['it'])
        self.writer.add_scalar('Loss/priv_reg_loss', locs['mean_priv_reg_loss'], locs['it'])
        self.writer.add_scalar('Loss/hist_latent_loss', locs['mean_hist_latent_loss'], locs['it'])
        self.writer.add_scalar('Loss/estimator_loss', locs['mean_estimator_loss'], locs['it'])
        self.writer.add_scalar('Loss/ss_loss', locs['mean_ss_loss'], locs['it'])
        self.writer.add_scalar('Loss/info_max_loss', locs['mean_info_max_loss'], locs['it'])
        self.writer.add_scalar('Loss/disc_loss', locs['mean_disc_loss'], locs['it'])
        self.writer.add_scalar('Loss/us_loss', locs['mean_us_loss'], locs['it'])
        self.writer.add_scalar('Loss/grad_pen_loss', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/disc_logit_loss', locs['mean_disc_logit_loss'], locs['it'])
        self.writer.add_scalar('Loss/disc_weight_decay', locs['mean_disc_weight_decay'], locs['it'])
        self.writer.add_scalar('Loss/mean_noise_std', mean_std.item(), locs['it'])

        self.writer.add_scalar('Acc/acc_pi', locs['mean_acc_pi'], locs['it'])
        self.writer.add_scalar('Acc/acc_exp', locs['mean_acc_exp'], locs['it'])
        self.writer.add_scalar('Acc/acc_lb', locs['mean_acc_lb'], locs['it'])
        self.writer.add_scalar('Acc/acc_ulb', locs['mean_acc_ulb'], locs['it'])

        self.writer.add_scalar('LR/lr_ac', self.alg.lr_ac, locs['it'])
        self.writer.add_scalar('LR/lr_disc', self.alg.lr_disc, locs['it'])
        self.writer.add_scalar('LR/lr_q', self.alg.lr_q, locs['it'])

        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rew_buffer']) > 0:
            mean_rewards = statistics.mean(locs['rew_buffer'])
            mean_rewards_i = statistics.mean(locs['rew_buffer_i'])
            mean_rewards_us = statistics.mean(locs['rew_buffer_us'])
            mean_rewards_ss = statistics.mean(locs['rew_buffer_ss'])
            mean_rewards_t = statistics.mean(locs['rew_buffer_t'])
            self.writer.add_scalar('Train/mean_reward', mean_rewards, locs['it'])
            self.writer.add_scalar('Train/mean_reward_i', mean_rewards_i, locs['it'])
            self.writer.add_scalar('Train/mean_reward_us', mean_rewards_us, locs['it'])
            self.writer.add_scalar('Train/mean_reward_ss', mean_rewards_ss, locs['it'])
            self.writer.add_scalar('Train/mean_reward_t', mean_rewards_t, locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['len_buffer']), locs['it'])

            pbar.set_description(
                f'mean_rewards: {mean_rewards:<6.4f}   '
                f'mean_rewards_i: {mean_rewards_i:<6.4f}   '
                f'mean_rewards_t: {mean_rewards_t:<6.4f}   '
                f'prior: {np.round(self.env.prior_prob.detach().cpu().numpy(), 4)}'
            )

    def save(self, path, infos=None):
        torch.save({
            'actor_critic': self.alg.actor_critic.state_dict(),
            'estimator': self.alg.estimator.state_dict(),
            'disc': self.alg.disc.state_dict(),
            'optim_ac': self.alg.optim_ac.state_dict(),
            'optim_hist_encoder': self.alg.optim_hist_encoder.state_dict(),
            'optim_estimator': self.alg.optim_estimator.state_dict(),
            'optim_d': self.alg.optim_d.state_dict(),
            'optim_q_eps': self.alg.optim_q_eps.state_dict(),
            'optim_q_c': self.alg.optim_q_c.state_dict(),
            'disc_normalizer': self.alg.disc_normalizer,
            'reward_i_normalizer': self.alg.disc.reward_i_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['actor_critic'])
        self.alg.estimator.load_state_dict(loaded_dict['estimator'])
        self.alg.disc.load_state_dict(loaded_dict['disc'])
        self.alg.disc_normalizer = loaded_dict['disc_normalizer']
        if loaded_dict['reward_i_normalizer']:
            self.alg.disc.reward_i_normalizer = loaded_dict['reward_i_normalizer']
        if load_optimizer:
            self.alg.optim_ac.load_state_dict(loaded_dict['optim_ac'])
            self.alg.optim_hist_encoder.load_state_dict(loaded_dict['optim_hist_encoder'])
            self.alg.optim_estimator.load_state_dict(loaded_dict['optim_estimator'])
            self.alg.optim_d.load_state_dict(loaded_dict['optim_d'])
            self.alg.optim_q_eps.load_state_dict(loaded_dict['optim_q_eps'])
            self.alg.optim_q_c.load_state_dict(loaded_dict['optim_q_c'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def load_pre_trained_actor(self, path_set):
        for i in range(len(path_set)):
            loaded_dict = torch.load(path_set[i], map_location=self.device)
            self.alg.pre_trained_actor_critics[i].load_state_dict(loaded_dict['actor_critic'], strict=False)
            self.alg.pre_trained_estimators[i].load_state_dict(loaded_dict['estimator'])

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
