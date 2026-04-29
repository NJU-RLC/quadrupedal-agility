import os
import sys
import time
import warnings
import statistics
import torch.nn.functional as F
import datetime
from copy import copy, deepcopy
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.algorithms import PPO, Discriminator
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
from rsl_rl.utils.utils import Normalizer


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu', **kwargs):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.device = device
        self.env = env

        self.num_obs = env.num_obs
        self.n_proprio = env.cfg.env.n_proprio
        self.n_delta_yaw = env.cfg.env.n_delta_yaw
        self.n_obst_type = env.cfg.env.n_obst_type
        self.n_auxiliary = env.cfg.env.n_auxiliary
        self.n_scan = env.cfg.env.n_scan
        self.n_priv = env.cfg.env.n_priv
        self.n_priv_latent = env.cfg.env.n_priv_latent
        self.history_len = env.cfg.env.history_len
        self.num_actions_d = env.num_actions_d
        self.num_actions_c = env.num_actions_c
        self.num_actions = 1 + self.num_actions_d * self.num_actions_c
        self.num_command = env.cfg.env.num_command

        self.num_obs_bbc = env.cfg.env.num_observations_bbc
        self.num_critic_obs = (env.cfg.env.num_observations_bbc + env.cfg.env.history_len *
                               (env.cfg.env.n_proprio - env.cfg.env.n_auxiliary))
        self.num_actions_bbc = env.cfg.env.num_actions_bbc

        self.num_disc_obs = env.cfg.env.num_obs_disc
        self.disc_obs_len = env.cfg.env.disc_obs_len
        self.disc_loss_function = train_cfg['runner']["disc_loss_function"]
        self.reward_i_coef = train_cfg['runner']['reward_i_coef']
        self.reward_us_coef = train_cfg['runner']['reward_us_coef']
        self.reward_ss_coef = train_cfg['runner']['reward_ss_coef']
        self.reward_t_coef = train_cfg['runner']['reward_t_coef']
        self.disc_hidden_units = train_cfg['runner']['disc_hidden_units']

        self.actor_critic: ActorCriticTSC = ActorCriticTSC(self.n_proprio,
                                                           self.n_auxiliary,
                                                           self.n_scan,
                                                           self.num_obs,
                                                           self.n_priv_latent,
                                                           self.n_priv,
                                                           self.history_len,
                                                           self.num_actions_d,
                                                           self.num_actions_c,
                                                           device=self.device,
                                                           **self.policy_cfg).to(self.device)

        self.actor_critic_bbc: ActorCriticBBC = ActorCriticBBC(self.num_obs_bbc,
                                                               self.num_critic_obs,
                                                               self.num_actions_bbc,
                                                               self.n_proprio,
                                                               self.n_auxiliary,
                                                               self.history_len,
                                                               self.n_priv,
                                                               self.n_priv_latent,
                                                               self.num_command,
                                                               **self.policy_cfg).to(self.device)

        self.estimator = Estimator(input_dim=self.n_proprio - self.n_auxiliary,
                                   output_dim=self.n_priv,
                                   hidden_dims=self.estimator_cfg["hidden_dims"]).to(self.device)
        # Depth encoder
        self.if_depth = self.depth_encoder_cfg["if_depth"]
        self.n_depth_latent = self.policy_cfg["scan_encoder_dims"][-1]
        if self.if_depth:
            self.depth_backbone = DepthOnlyFCBackbone58x87(self.n_proprio,
                                                           self.n_depth_latent,
                                                           self.depth_encoder_cfg["hidden_dims"],
                                                           )
            self.depth_encoder = RecurrentDepthBackbone(self.depth_backbone, self.n_depth_latent, env.cfg).to(
                self.device)
            self.depth_actor = deepcopy(self.actor_critic.actor)
            self.depth_backbone.augment = self.depth_encoder.byol_learner.augment1
        else:
            self.depth_encoder = None
            self.depth_actor = None

        # Create algorithm
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(self.actor_critic, self.actor_critic_bbc,
                                  self.estimator, self.estimator_cfg,
                                  self.depth_encoder, self.depth_encoder_cfg, self.depth_actor,
                                  device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.num_actions],
        )

        disc_normalizer = Normalizer(self.num_disc_obs * self.disc_obs_len)
        reward_i_normalizer = None
        if self.disc_loss_function == "WassersteinLoss":
            reward_i_normalizer = Normalizer(1)
        self.discriminator = Discriminator(
            self.num_disc_obs * self.disc_obs_len,
            self.num_disc_obs,
            self.env.dim_c,
            self.env.dt,
            self.disc_loss_function,
            reward_i_normalizer,
            self.reward_i_coef,
            self.reward_us_coef,
            self.reward_ss_coef,
            self.reward_t_coef,
            self.disc_obs_len,
            self.disc_hidden_units,
            disc_normalizer, device).to(self.device)

        self.learn = self.learn_RL if not self.if_depth else self.learn_vision

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_estimator_loss = 0.
        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0.
        priv_reg_coef = 0.
        entropy_coef = 0.
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        obs_bbc = self.env.get_observations_bbc()
        disc_obs = self.env.get_observations_disc()
        obs_disc_history_buf = torch.stack([disc_obs] * self.disc_obs_len, dim=1)
        action_history_buf = torch.zeros(self.env.num_envs, self.env.cfg.domain_rand.action_buf_len, self.num_actions,
                                         device=self.device, dtype=torch.float)
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, obs_bbc, critic_obs = obs.to(self.device), obs_bbc.to(self.device), critic_obs.to(self.device)
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rew_buffer = deque(maxlen=1000)
        rew_buffer_i = deque(maxlen=1000)
        rew_buffer_us = deque(maxlen=1000)
        rew_buffer_ss = deque(maxlen=1000)
        rew_buffer_t = deque(maxlen=1000)
        len_buffer = deque(maxlen=1000)
        reach_goal_buffer = deque(maxlen=1000)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_sum_i = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_sum_us = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_sum_ss = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_sum_t = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding=hist_encoding)
                    action_history_buf = torch.cat([action_history_buf[:, 1:].clone(), actions[:, None, :].clone()],
                                                   dim=1)
                    next_commands = self.env.set_commands(actions)
                    obs_bbc[:, -next_commands.shape[1]:] = next_commands
                    actions_bbc = self.alg.actor_critic_bbc.act_inference(obs_bbc, hist_encoding=True).detach()
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_disc_states = self.env.step(
                        actions_bbc, action_history_buf)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    next_obs_bbc = self.env.get_observations_bbc()
                    disc_obs = self.env.get_observations_disc()
                    obs, critic_obs, next_obs_bbc, disc_obs, rewards, dones = obs.to(self.device), \
                        critic_obs.to(self.device), next_obs_bbc.to(self.device), disc_obs.to(self.device), \
                        rewards.to(self.device), dones.to(self.device)

                    disc_obs_with_term = torch.clone(disc_obs)
                    disc_obs_with_term[reset_env_ids] = terminal_disc_states
                    obs_disc_history_buf = torch.cat([obs_disc_history_buf[:, 1:],
                                                      disc_obs_with_term.unsqueeze(1)], dim=1)

                    rewards, reward_i, reward_us, reward_ss, reward_t = self.discriminator.predict_disc_reward(
                        rewards.unsqueeze(1), obs_bbc, obs_disc_history_buf)
                    total_rew = self.alg.process_env_step(rewards, dones, infos)

                    obs_bbc = torch.clone(next_obs_bbc)
                    obs_disc_history_buf[reset_env_ids] = torch.stack([disc_obs[reset_env_ids]] *
                                                                      self.disc_obs_len, dim=1)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_reward_sum_i += reward_i
                        cur_reward_sum_us += reward_us
                        cur_reward_sum_ss += reward_ss
                        cur_reward_sum_t += reward_t
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        rew_buffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_buffer_i.extend(cur_reward_sum_i[new_ids][:, 0].cpu().numpy().tolist())
                        rew_buffer_us.extend(cur_reward_sum_us[new_ids][:, 0].cpu().numpy().tolist())
                        rew_buffer_ss.extend(cur_reward_sum_ss[new_ids][:, 0].cpu().numpy().tolist())
                        rew_buffer_t.extend(cur_reward_sum_t[new_ids][:, 0].cpu().numpy().tolist())
                        len_buffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        reach_goal_buffer.extend(infos['reach_goal'][new_ids][:, 0].cpu().numpy().tolist())

                        cur_reward_sum[new_ids] = 0
                        cur_reward_sum_i[new_ids] = 0
                        cur_reward_sum_us[new_ids] = 0
                        cur_reward_sum_ss[new_ids] = 0
                        cur_reward_sum_t[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            (mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc,
             mean_priv_reg_loss, priv_reg_coef) = self.alg.update()
            if hist_encoding:
                print("Updating dagger...")
                mean_hist_latent_loss = self.alg.update_dagger()

            stop = time.time()
            learn_time = stop - start
            if len(reach_goal_buffer) > 0:
                self.env.success_rate = statistics.mean(reach_goal_buffer)
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model.pt'))
            ep_infos.clear()

    def learn_vision(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.env.cfg.env.draw_height_maps = False
        self.env.cfg.noise.add_noise = False
        self.env.cfg.obstacle.curriculum = False

        self.env.cfg.env.next_goal_threshold = 0.45
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        ep_infos = []
        rew_buffer = deque(maxlen=1000)
        len_buffer = deque(maxlen=1000)
        reach_goal_buffer = deque(maxlen=1000)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        action_student_history_buf = torch.zeros(self.env.num_envs, self.env.cfg.domain_rand.action_buf_len,
                                                 self.num_actions, device=self.device, dtype=torch.float)

        obs = self.env.get_observations()
        obs_bbc = self.env.get_observations_bbc()
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1] if self.if_depth else None
        infos["delta_yaw_ok"] = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.alg.depth_encoder.train()
        self.alg.depth_actor.train()

        num_pretrain_iter = 0
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            depth_buffer = []
            depth_latent_buffer = []
            scandots_latent_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []
            yaw_buffer_student = []
            yaw_buffer_teacher = []
            obst_type_buffer_student = []
            obst_type_buffer_teacher = []
            delta_yaw_ok_buffer = []
            for i in range(self.depth_encoder_cfg["num_steps_per_env"]):
                if infos["depth"] != None:
                    with torch.no_grad():
                        priv_states_estimated = self.alg.estimator(obs[:, :self.alg.num_prop])
                        obs[:,
                        self.alg.num_prop + self.alg.num_auxiliary + self.alg.num_scan:self.alg.num_prop + self.alg.num_auxiliary + self.alg.num_scan +
                                                                                       self.alg.priv_states_dim] = priv_states_estimated
                        scandots_latent = self.alg.actor_critic.actor.infer_scandots_latent(obs)
                    scandots_latent_buffer.append(scandots_latent)
                    obs_prop_depth = obs[:, :self.n_proprio].clone()
                    obs_prop_depth[:,
                    self.n_proprio - self.n_auxiliary:self.n_proprio] = 0  # mask delta_yaws, obstacle_types
                    depth_latent_yaw_type = self.alg.depth_encoder(infos["depth"].clone(),
                                                                   obs_prop_depth)  # clone is crucial to avoid in-place operation

                    depth_latent = depth_latent_yaw_type[:, :self.n_depth_latent]
                    delta_yaw = 1.5 * depth_latent_yaw_type[:,
                                      self.n_depth_latent:self.n_depth_latent + self.n_delta_yaw]
                    obst_type = depth_latent_yaw_type[:, self.n_depth_latent + self.n_delta_yaw:]
                    depth_buffer.append(infos["depth"].clone())
                    depth_latent_buffer.append(depth_latent)
                    yaw_buffer_student.append(delta_yaw)
                    yaw_buffer_teacher.append(
                        obs[:, self.n_proprio - self.n_auxiliary:self.n_proprio - self.n_auxiliary + self.n_delta_yaw])
                    obst_type_buffer_student.append(obst_type)
                    obst_type_buffer_teacher.append(obs[:, self.n_proprio - self.n_auxiliary +
                                                           self.n_delta_yaw:self.n_proprio])

                with torch.no_grad():
                    actions_teacher = self.alg.actor_critic.act_inference(obs, hist_encoding=True, scandots_latent=None)
                    actions_teacher_buffer.append(actions_teacher)

                obs_student = obs.clone()
                obs_student[infos["delta_yaw_ok"],
                self.n_proprio - self.n_auxiliary:self.n_proprio - self.n_auxiliary + self.n_delta_yaw] \
                    = delta_yaw.detach()[infos["delta_yaw_ok"]]
                obs_student[:, self.n_proprio - self.n_auxiliary + self.n_delta_yaw:self.n_proprio] = \
                    F.one_hot(torch.argmax(obst_type.detach(), dim=-1), num_classes=obst_type.shape[-1])
                delta_yaw_ok_buffer.append(torch.nonzero(infos["delta_yaw_ok"]).size(0) / infos["delta_yaw_ok"].numel())
                embedding = self.alg.depth_actor(obs_student, hist_encoding=True, scandots_latent=depth_latent)
                prob = self.depth_actor.actor_d(embedding)
                mean = self.depth_actor.actor_c(embedding)
                actions_student = torch.cat([torch.argmax(prob, dim=-1).unsqueeze(-1), mean], dim=-1)
                actions_student_buffer.append(torch.cat([prob, mean], dim=-1))

                action_student_history_buf = torch.cat([action_student_history_buf[:, 1:].clone(),
                                                        actions_student[:, None, :].clone()], dim=1)
                actions_student = action_student_history_buf[:, -1]

                # detach actions before feeding the env
                if it < num_pretrain_iter:
                    next_commands = self.env.set_commands(actions_teacher.detach())
                else:
                    next_commands = self.env.set_commands(actions_student.detach())
                obs_bbc[:, -next_commands.shape[1]:] = next_commands
                actions_bbc = self.alg.actor_critic_bbc.act_inference(obs_bbc, hist_encoding=True).detach()
                obs, privileged_obs, rewards, dones, infos, _, _ = self.env.step(actions_bbc)
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs_bbc = self.env.get_observations_bbc()
                obs, critic_obs, obs_bbc, rewards, dones = (obs.to(self.device), critic_obs.to(self.device),
                                                            obs_bbc.to(self.device), rewards.to(self.device),
                                                            dones.to(self.device))
                action_student_history_buf[(dones > 0).nonzero(as_tuple=False)] *= 0

                if self.log_dir is not None:
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rew_buffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    len_buffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    reach_goal_buffer.extend(infos['reach_goal'][new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
            scandots_latent_buffer = torch.cat(scandots_latent_buffer, dim=0)
            depth_latent_buffer = torch.cat(depth_latent_buffer, dim=0)

            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
            yaw_buffer_student = torch.cat(yaw_buffer_student, dim=0)
            yaw_buffer_teacher = torch.cat(yaw_buffer_teacher, dim=0)
            obst_type_buffer_student = torch.cat(obst_type_buffer_student, dim=0)
            obst_type_buffer_teacher = torch.cat(obst_type_buffer_teacher, dim=0)
            depth_buffer = torch.cat(depth_buffer, dim=0)
            depth_actor_loss, yaw_loss, obst_type_loss, byol_loss = self.alg.update_depth_actor(actions_student_buffer,
                                                                                                actions_teacher_buffer,
                                                                                                yaw_buffer_student,
                                                                                                yaw_buffer_teacher,
                                                                                                obst_type_buffer_student,
                                                                                                obst_type_buffer_teacher,
                                                                                                depth_buffer)

            stop = time.time()
            learn_time = stop - start

            self.alg.depth_encoder.detach_hidden_states()

            # learning rate decay
            for param_group in self.alg.depth_encoder_optimizer.param_groups:
                param_group['lr'] = max(self.depth_encoder_cfg["learning_rate"] - (
                        self.depth_encoder_cfg["learning_rate"] - self.depth_encoder_cfg[
                    "learning_rate_min"]) * it / 20000, self.depth_encoder_cfg["learning_rate_min"])
            for param_group in self.alg.depth_actor_optimizer.param_groups:
                param_group['lr'] = max(self.depth_encoder_cfg["learning_rate"] - (
                        self.depth_encoder_cfg["learning_rate"] - self.depth_encoder_cfg[
                    "learning_rate_min"]) * it / 20000, self.depth_encoder_cfg["learning_rate_min"])
            for param_group in self.alg.byol_optimizer.param_groups:
                param_group['lr'] = max(self.depth_encoder_cfg["learning_rate_byol"] - (
                        self.depth_encoder_cfg["learning_rate_byol"] - self.depth_encoder_cfg[
                    "learning_rate_min"]) * it / 20000, self.depth_encoder_cfg["learning_rate_min"])

            if self.log_dir is not None:
                self.log_vision(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model.pt'.format(it)))
            ep_infos.clear()

    def log_vision(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        log_dict = {}
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
                value = torch.mean(infotensor)
                log_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        log_dict['Loss_depth/delta_yaw_ok_percent'] = locs['delta_yaw_ok_percentage']
        log_dict['Loss_depth/depth_actor'] = locs['depth_actor_loss']
        log_dict['Loss_depth/yaw'] = locs['yaw_loss']
        log_dict['Loss_depth/obst_type'] = locs['obst_type_loss']
        log_dict['Loss_depth/byol'] = locs['byol_loss']
        log_dict['Policy/mean_noise_std'] = mean_std.item()
        log_dict['Perf/total_fps'] = fps
        log_dict['Perf/collection time'] = locs['collection_time']
        log_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rew_buffer']) > 0:
            log_dict['Train/mean_reward'] = statistics.mean(locs['rew_buffer'])
            log_dict['Train/mean_episode_length'] = statistics.mean(locs['len_buffer'])
            log_dict['Train/mean_success_rate'] = statistics.mean(locs['reach_goal_buffer'])

        for log_dict_key, log_dict_value in log_dict.items():
            self.writer.add_scalar(log_dict_key, log_dict_value, locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rew_buffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rew_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['len_buffer']):.2f}\n"""
                          f"""{'Mean success rate:':>{pad}} {statistics.mean(locs['reach_goal_buffer']):.2f}\n"""
                          f"""{'Depth actor loss:':>{pad}} {locs['depth_actor_loss']:.4f}\n"""
                          f"""{'Yaw loss:':>{pad}} {locs['yaw_loss']:.4f}\n"""
                          f"""{'Delta yaw ok percentage:':>{pad}} {locs['delta_yaw_ok_percentage']:.4f}\n""")
        else:
            log_string = (f"""{'#' * width}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        hours, mins, secs = self.s_to_hms(self.tot_time)
        hours_eta, mins_eta, secs_eta = self.s_to_hms(eta)
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {hours:.0f}h {mins:.0f}mins {secs:.0f}secs\n"""
                       f"""{'ETA:':>{pad}} {hours_eta:.0f}h {mins_eta:.0f}mins {secs_eta:.0f}secs\n""")
        print(log_string)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        log_dict = {}
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
                value = torch.mean(infotensor)
                log_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        log_dict['Loss/value_function'] = locs['mean_value_loss']
        log_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        log_dict['Loss/estimator'] = locs['mean_estimator_loss']
        log_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        log_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        log_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
        log_dict['Loss/entropy_coef'] = locs['entropy_coef']
        log_dict['Loss/learning_rate'] = self.alg.learning_rate
        log_dict['Loss/discriminator'] = locs['mean_disc_loss']
        log_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']

        log_dict['Policy/mean_noise_std'] = mean_std.item()
        log_dict['Curr/bar_jump_bias'] = self.env.bar_jump_bias
        log_dict['Curr/tire_jump_bias'] = self.env.tire_jump_bias
        log_dict['Perf/total_fps'] = fps
        log_dict['Perf/collection time'] = locs['collection_time']
        log_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rew_buffer']) > 0:
            log_dict['Train/mean_reward'] = statistics.mean(locs['rew_buffer'])
            log_dict['Train/mean_reward_i'] = statistics.mean(locs['rew_buffer_i'])
            log_dict['Train/mean_reward_us'] = statistics.mean(locs['rew_buffer_us'])
            log_dict['Train/mean_reward_ss'] = statistics.mean(locs['rew_buffer_ss'])
            log_dict['Train/mean_reward_t'] = statistics.mean(locs['rew_buffer_t'])
            log_dict['Train/mean_episode_length'] = statistics.mean(locs['len_buffer'])
            log_dict['Train/mean_success_rate'] = statistics.mean(locs['reach_goal_buffer'])

        for log_dict_key, log_dict_value in log_dict.items():
            self.writer.add_scalar(log_dict_key, log_dict_value, locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rew_buffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                          f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rew_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['len_buffer']):.2f}\n"""
                          f"""{'Mean success rate:':>{pad}} {statistics.mean(locs['reach_goal_buffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        hours, mins, secs = self.s_to_hms(self.tot_time)
        hours_eta, mins_eta, secs_eta = self.s_to_hms(eta)
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {hours:.0f}h {mins:.0f}mins {secs:.0f}secs\n"""
                       f"""{'ETA:':>{pad}} {hours_eta:.0f}h {mins_eta:.0f}mins {secs_eta:.0f}secs\n""")
        print(log_string)

    @staticmethod
    def s_to_hms(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        return hours, minutes, remaining_seconds

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if self.if_depth:
            state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if self.if_depth:
            if 'depth_encoder_state_dict' not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
            if 'depth_actor_state_dict' in loaded_dict:
                print("Saved depth actor detected, loading...")
                self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def load_bbc(self, path):
        print("*" * 80)
        print("Loading bbc model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic_bbc.load_state_dict(loaded_dict['actor_critic'])
        if self.estimator_cfg['load_estimator_bbc']:
            self.alg.estimator.load_state_dict(loaded_dict['estimator'])
        self.discriminator.load_state_dict(loaded_dict['disc'])
        self.discriminator.normalizer = loaded_dict['disc_normalizer']
        if loaded_dict['reward_i_normalizer']:
            self.discriminator.reward_i_normalizer = loaded_dict['reward_i_normalizer']
        self.alg.actor_critic_bbc.eval()
        self.discriminator.eval()
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_policy_bbc(self, device=None):
        self.alg.actor_critic_bbc.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic_bbc.to(device)
        return self.alg.actor_critic_bbc.act_inference

    def get_depth_actor_inference_policy(self, device=None):
        self.alg.depth_actor.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.depth_actor.to(device)
        return self.alg.depth_actor

    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic

    def get_estimator_inference_policy(self, device=None):
        self.alg.estimator.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator.inference

    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder

    def get_disc_inference_policy(self, device=None):
        self.alg.discriminator.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.discriminator.to(device)
        return self.alg.discriminator.inference
