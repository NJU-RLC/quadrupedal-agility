from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import copy
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.obstacle import Obstacle
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.torch_jit_utils import *
from scipy.spatial.transform import Rotation as R
from .legged_robot_config import LeggedRobotCfg

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

KEY_BODY_NAMES = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']


def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:, 0];
    y = quat_angle[:, 1];
    z = quat_angle[:, 2];
    w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self.mocap_category = self.cfg.env.mocap_category_all
        self.dim_c = len(self.mocap_category)
        self._parse_cfg(self.cfg)
        self.success_rate = 0
        self.obst_curr_count = 0
        self.bar_jump_bias = 0
        self.tire_jump_bias = 0
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]),
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0
        self.key_body_ids = self.build_key_body_ids_tensor(self.env_handle, self.actor_handle)

        # Create a mapping dictionary
        self.category_mapping = {category: index for index, category in enumerate(self.cfg.env.mocap_category_all)}
        # Convert the elements in mocap_category to their corresponding indexes
        self.mocap_indices = torch.tensor([self.category_mapping[category] for category in self.cfg.env.mocap_category],
                                          device=self.device)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()

    def step(self, actions, action_hl_history_buf=None):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self.reindex(actions)  # [FL, FR, RL, RR] -> [FR, FL, RR, RL]
        actions.to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()],
                                            dim=1)
        self.action_hl_history_buf = action_hl_history_buf
        if self.cfg.domain_rand.action_delay:
            indices = -self.cfg.domain_rand.action_delay_step - 1
            actions = self.action_history_buf[:, indices]  # delay for 1/50=20ms

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            if self.obstacle:
                torques = torch.cat([self.torques.flatten(), self.torques_obst.flatten()], dim=0)
            else:
                torques = self.torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        reset_env_ids, terminal_disc_states = self.post_physics_step()
        # clip_obs = self.cfg.normalization.clip_observations
        # self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # if self.privileged_obs_buf is not None:
        #     self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.extras["delta_yaw_ok"] = torch.abs(self.delta_yaw) < 0.6
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_disc_states

    def get_history_observations(self):
        return self.obs_history_buf

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (
                self.cfg.depth.far_clip - self.cfg.depth.near_clip) - 0.5
        return depth_image

    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        # depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)

        depth_noise = self.cfg.depth.depth_noise * torch.rand(1)[0]
        depth_image += self.cfg.depth.depth_noise * 2 * (torch.rand(1) - 0.5)[0]
        depth_image += depth_noise * 2 * (torch.rand_like(depth_image) - 0.5)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[1:-1, 10:-9]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim)  # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
                                                                self.envs[i],
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)

            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)],
                                                 dim=0)

        self.gym.end_access_image_tensors(self.sim)

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2],
                                           dim=1) < self.cfg.env.next_goal_threshold
        self.leave_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2],
                                         dim=1) > self.cfg.env.leave_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        # self._update_jump_schedule()
        self._update_goals()
        self._post_physics_step_callback()

        cur_goal_idx = torch.clamp(self.cur_goal_idx, 0, self.env_goals.size(1) - self.obstacle.last_goal_repeat - 1)
        cur_goal_idx = F.one_hot(cur_goal_idx // self.obstacle.num_goals,
                                 num_classes=self.obstacle_types.size(1)).to(torch.bool)
        self.cur_obstacle_types = self.obstacle_types[cur_goal_idx]

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_disc_states = self.get_observations_disc()[env_ids]
        self.reset_idx(env_ids)

        # if self.obstacle:
        #     joint_idx = self.obst_dof_idx.view(self.num_envs, -1).cpu().numpy()
        #     self.obstacle.obstacle_joint_pos[joint_idx] = self.obst_dof_pos.cpu().numpy()[:, 0]
        #     self.obstacle.refresh_seesaw()
        #     self._refresh_obstacle()
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.update_depth_buffer()

        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques_org[:] = self.torques_org[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            if not self.cfg.depth.use_camera:
                self.gym.clear_lines(self.viewer)
                self._draw_height_samples()
                self._draw_goals()
                self._draw_feet()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                image_color = (self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5) * 255
                image_color = cv2.applyColorMap(cv2.convertScaleAbs(image_color), cv2.COLORMAP_JET)
                cv2.imshow("Depth Image", image_color)
                cv2.waitKey(1)

        return env_ids, terminal_disc_states

    def reindex_feet(self, vec):
        # [FL, FR, RL, RR] -> [FR, FL, RR, RL]
        # return vec[:, [1, 0, 3, 2]]
        # [FL, FR, RL, RR]
        return vec[:, [0, 1, 2, 3]]

    def reindex(self, vec):
        # [FL, FR, RL, RR] -> [FR, FL, RR, RL]
        # return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]
        # [FL, FR, RL, RR]
        return vec[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

    def build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        roll_cutoff = torch.abs(self.roll) > 1.5
        pitch_cutoff = torch.abs(self.pitch) > 1.5
        reach_goal_cutoff = self.cur_goal_idx >= (self.env_goals.size(1) - self.obstacle.last_goal_repeat)
        height_cutoff = self.root_states[:, 2] < -0.25
        leave_goal_cutoff = self.leave_goal_ids
        reach_last_goal = torch.norm(self.root_states[:, :2] - self.env_goals[:, -self.obstacle.last_goal_repeat, :2],
                                     dim=1) < self.cfg.env.next_goal_threshold

        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.time_out_buf |= reach_goal_cutoff

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff
        self.reset_buf |= leave_goal_cutoff
        if self.cfg.depth.use_camera:
            self.reset_buf |= reach_last_goal
        self.extras["reach_goal"] = reach_goal_cutoff

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.obstacle.curriculum and self.cfg.terrain.mesh_type == "obstacle":
            self._update_obstacle_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        if self.obstacle.cfg.randomize_start:
            self.cur_obst_idx[env_ids] = torch.randint_like(self.cur_goal_idx, 0, self.obstacle.env_goals.shape[1])[
                env_ids]
            self.cur_goal_idx[env_ids] = self.cur_obst_idx[env_ids] * self.obstacle.env_goals.shape[2]
            self.root_init_pos[env_ids] = (self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]).
                                                                 expand(-1, -1, self.env_goals.shape[-1])).squeeze(1))[
                env_ids]
            self.root_init_ang[env_ids] = (self.obst_angs.gather(1, (self.cur_obst_idx[:, None])).squeeze(1))[env_ids]
        else:
            self.cur_goal_idx[env_ids] = 0

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        # self._resample_commands(env_ids)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques_org[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ 
        Computes observations
        """
        if self.cfg.obstacle.measure_heights:
            root_h = (self.root_states[:, 2] -
                      self.measured_heights[:, self.measured_heights.shape[1] // 2 + 1]).view(-1, 1)
        else:
            root_h = self.root_states[:, 2].view(-1, 1)
        if self.cfg.env.root_height_obs:
            root_h_obs = root_h
        else:
            root_h_obs = torch.zeros_like(root_h)
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        if self.global_counter % self.cfg.depth.update_interval == 0:
            self.delta_yaw = self.target_yaw - self.yaw
            self.delta_next_yaw = self.next_target_yaw - self.yaw
            # Normalize to [-pi, pi]
            self.delta_yaw = (self.delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi
            self.delta_next_yaw = (self.delta_next_yaw + torch.pi) % (2 * torch.pi) - torch.pi

        delta_yaws = torch.cat([self.delta_yaw[:, None], self.delta_next_yaw[:, None]], dim=-1)

        key_body_pos = self.rigid_body_pos[:, self.key_body_ids, :]
        flat_local_key_pos = compute_flat_key_pos(self.root_states, key_body_pos)

        self.obs_disc_buf = torch.cat([imu_obs, root_h, self.base_lin_vel * self.obs_scales.lin_vel_dist,
                                       self.base_ang_vel * self.obs_scales.ang_vel_dist,
                                       (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                       self.dof_vel * self.obs_scales.dof_vel,
                                       flat_local_key_pos * self.obs_scales.key_pos,
                                       self.contact_filt.float() * self.obs_scales.foot_contact], dim=-1)

        obs_buf = torch.cat([imu_obs,
                             self.base_ang_vel * self.obs_scales.ang_vel,
                             (self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos,
                             self.dof_vel * self.obs_scales.dof_vel,
                             self.action_history_buf[:, -1],
                             self.contact_filt.float() - 0.5,
                             flat_local_key_pos * 0], dim=-1)

        priv_explicit = torch.cat([root_h_obs, self.base_lin_vel * self.obs_scales.lin_vel], dim=-1)

        priv_latent = torch.cat((
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1,
            self.motor_strength[1] - 1
        ), dim=-1)
        obstacle_types = F.one_hot(self.cur_obstacle_types, num_classes=len(self.obstacle.proportions))
        if self.cfg.obstacle.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            self.obs_buf = torch.cat(
                [obs_buf, delta_yaws,
                 obstacle_types, heights, priv_explicit,
                 priv_latent,
                 self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat(
                [obs_buf, delta_yaws,
                 obstacle_types, priv_explicit, priv_latent,
                 self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        # bbc observation
        self.obs_bbc_buf = torch.cat([obs_buf, priv_explicit, priv_latent], dim=-1)
        self.obs_bbc_buf = torch.cat((self.obs_bbc_buf, self.obs_history_buf.view(self.num_envs, -1)), dim=-1)
        self.obs_bbc_buf = torch.cat([self.obs_bbc_buf, self.commands, self.latent_eps, self.latent_c], dim=-1)

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.contact_buf = torch.cat([self.contact_buf[:, 1:], self.contact_filt.float().unsqueeze(1)], dim=1)

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.obs_bbc_buf = torch.clip(self.obs_bbc_buf, -clip_obs, clip_obs)
        self.obs_history_buf = torch.clip(self.obs_history_buf, -clip_obs, clip_obs)
        self.contact_buf = torch.clip(self.contact_buf, -clip_obs, clip_obs)

    def get_observations_disc(self):
        return self.obs_disc_buf

    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            x = x + (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
        return x

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        start = time()
        print("*" * 80)
        print("Start creating ground...")
        self.terrain = None
        self.obstacle = None
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        elif mesh_type == 'obstacle':
            self.obstacle = Obstacle(self.cfg.obstacle, self.num_envs)
        if mesh_type in ['plane', 'obstacle']:
            self._create_ground_plane()
            if mesh_type == 'obstacle':
                self._create_obstacle()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*" * 80)
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1,))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1,))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3,))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        if self.cfg.obstacle.measure_heights:
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None] + future).expand(-1, -1, self.env_goals.shape[
            -1])).squeeze(1)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, :] = 0.0  # refresh command

        lin_vel_x = torch.tensor(self.command_ranges["lin_vel_x"], device=self.device)
        lin_vel_y = torch.tensor(self.command_ranges["lin_vel_y"], device=self.device)
        ang_vel_yaw = torch.tensor(self.command_ranges["ang_vel_yaw"], device=self.device)
        lin_vel_x_l, lin_vel_x_h = (lin_vel_x[:, 0].unsqueeze(0).repeat(len(env_ids), 1),
                                    lin_vel_x[:, 1].unsqueeze(0).repeat(len(env_ids), 1))
        lin_vel_y_l, lin_vel_y_h = (lin_vel_y[:, 0].unsqueeze(0).repeat(len(env_ids), 1),
                                    lin_vel_y[:, 1].unsqueeze(0).repeat(len(env_ids), 1))
        ang_vel_yaw_l, ang_vel_yaw_h = (ang_vel_yaw[:, 0].unsqueeze(0).repeat(len(env_ids), 1),
                                        ang_vel_yaw[:, 1].unsqueeze(0).repeat(len(env_ids), 1))

        latent_c_idx = torch.argmax(self.latent_c[env_ids], dim=-1)

        lin_vel_x_l = lin_vel_x_l[torch.arange(len(latent_c_idx)), latent_c_idx]
        lin_vel_x_h = lin_vel_x_h[torch.arange(len(latent_c_idx)), latent_c_idx]
        lin_vel_y_l = lin_vel_y_l[torch.arange(len(latent_c_idx)), latent_c_idx]
        lin_vel_y_h = lin_vel_y_h[torch.arange(len(latent_c_idx)), latent_c_idx]
        ang_vel_yaw_l = ang_vel_yaw_l[torch.arange(len(latent_c_idx)), latent_c_idx]
        ang_vel_yaw_h = ang_vel_yaw_h[torch.arange(len(latent_c_idx)), latent_c_idx]

        self.commands[env_ids, 0] = torch_rand_floats(lin_vel_x_l.unsqueeze(1), lin_vel_x_h.unsqueeze(1),
                                                      (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_floats(lin_vel_y_l.unsqueeze(1), lin_vel_y_h.unsqueeze(1),
                                                      (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_floats(ang_vel_yaw_l.unsqueeze(1), ang_vel_yaw_h.unsqueeze(1),
                                                      (len(env_ids), 1), device=self.device).squeeze(1)

        jump_command = (latent_c_idx == (self.dim_c - 1))

        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["jump_height"][0],
                                                     self.command_ranges["jump_height"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1) * (jump_command.float())

        self.commands[env_ids, 4] = torch_rand_float(self.command_ranges["locomotion_height"][0],
                                                     self.command_ranges["locomotion_height"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1) * ((~jump_command).float())

        # # set small commands to zero
        # self.commands[env_ids, 0] *= (torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip)
        # self.commands[env_ids, 1] *= (torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_y_clip)
        # self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip)

    def set_commands(self, actions):
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        actions_d = actions[:, 0].to(torch.long)
        # Map the index values in actions_d to the index values in mocap_indice
        mapped_actions_d = self.mocap_indices[actions_d]
        actions_c = actions[
            torch.arange(actions.size(0), device=self.device)[:, None],
            actions_d[:, None] * self.num_actions_c + torch.arange(self.num_actions_c, device=self.device) + 1]

        if len(env_ids):
            mapped_actions_d = mapped_actions_d[env_ids]
            commands = torch.clip(actions_c[env_ids, :], -1, 1)

            self.latent_c[env_ids, :] = 0
            self.latent_c[env_ids, mapped_actions_d] = 1

            self.latent_eps[env_ids, 0] = commands[:, -1].clone()

            commands = (commands + 1) / 2

            self.commands[env_ids, :] = 0.0  # refresh command

            lin_vel_x = torch.tensor(self.command_ranges["lin_vel_x"], device=self.device)
            lin_vel_y = torch.tensor(self.command_ranges["lin_vel_y"], device=self.device)
            ang_vel_yaw = torch.tensor(self.command_ranges["ang_vel_yaw"], device=self.device)
            lin_vel_x_l, lin_vel_x_h = (lin_vel_x[:, 0].unsqueeze(0).repeat(len(env_ids), 1),
                                        lin_vel_x[:, 1].unsqueeze(0).repeat(len(env_ids), 1))
            lin_vel_y_l, lin_vel_y_h = (lin_vel_y[:, 0].unsqueeze(0).repeat(len(env_ids), 1),
                                        lin_vel_y[:, 1].unsqueeze(0).repeat(len(env_ids), 1))
            ang_vel_yaw_l, ang_vel_yaw_h = (ang_vel_yaw[:, 0].unsqueeze(0).repeat(len(env_ids), 1),
                                            ang_vel_yaw[:, 1].unsqueeze(0).repeat(len(env_ids), 1))

            lin_vel_x_l = lin_vel_x_l[torch.arange(len(env_ids)), mapped_actions_d]
            lin_vel_x_h = lin_vel_x_h[torch.arange(len(env_ids)), mapped_actions_d]
            lin_vel_y_l = lin_vel_y_l[torch.arange(len(env_ids)), mapped_actions_d]
            lin_vel_y_h = lin_vel_y_h[torch.arange(len(env_ids)), mapped_actions_d]
            ang_vel_yaw_l = ang_vel_yaw_l[torch.arange(len(env_ids)), mapped_actions_d]
            ang_vel_yaw_h = ang_vel_yaw_h[torch.arange(len(env_ids)), mapped_actions_d]

            self.commands[env_ids, 0] = lin_vel_x_l + (lin_vel_x_h - lin_vel_x_l) * commands[:, 0]
            self.commands[env_ids, 1] = lin_vel_y_l + (lin_vel_y_h - lin_vel_y_l) * commands[:, 1]
            self.commands[env_ids, 2] = ang_vel_yaw_l + (ang_vel_yaw_h - ang_vel_yaw_l) * commands[:, 2]

            jump_height_l = self.command_ranges["jump_height"][0]
            jump_height_h = self.command_ranges["jump_height"][1]
            locomotion_height_l = self.command_ranges["locomotion_height"][0]
            locomotion_height_h = self.command_ranges["locomotion_height"][1]

            jump_command = (mapped_actions_d == (self.dim_c - 1))

            self.commands[env_ids, 3] = (jump_height_l + (jump_height_h - jump_height_l) *
                                         commands[:, 3]) * jump_command.float()
            self.commands[env_ids, 4] = (locomotion_height_l + (locomotion_height_h - locomotion_height_l) *
                                         commands[:, 4]) * (~jump_command).float()

        if self.cfg.domain_rand.randomize_action:
            action_noise = torch_rand_float(self.cfg.domain_rand.action_noise[0], self.cfg.domain_rand.action_noise[1],
                                            (self.commands.shape[0], self.commands.shape[1]), device=self.device)
            self.commands *= action_noise
        next_commands = torch.cat([self.commands, self.latent_eps, self.latent_c], dim=-1)
        return next_commands

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range
        control_type = self.cfg.control.control_type
        if control_type == "P":
            if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
                torques = self.p_gains * (
                        actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains * self.dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains * (
                        actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[
                              1] * self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (
                    self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        if self.obstacle:
            self.torques_obst = (self.obst_stiffness * (self.target_obst_dof_pos[:, 0] - self.obst_dof_pos[:, 0]) -
                                 self.obst_damping * self.obst_dof_vel[:, 0])
        self.torques_org = torques
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos
        # self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.9, (len(env_ids), self.num_dof),
        #                                                                 device=self.device)
        self.dof_vel[env_ids] = 0.

        if self.obstacle:
            # reset seesaw
            bar_jump_idx = (torch.flatten(self.obstacle_types) == 0)
            seesaw_idx = (torch.flatten(self.obstacle_types) == 3)
            tire_jump_idx = (torch.flatten(self.obstacle_types) == 4)
            obst_dof_idx = (bar_jump_idx | seesaw_idx | tire_jump_idx)
            obst_idx = torch.flatten(self.obstacle_types)[obst_dof_idx]
            if self.obstacle.cfg.randomize_start:
                seesaw_passed = (self.cur_obst_idx > (self.obstacle_types == 3).int().argmax(dim=1))[env_ids]
                reset_seesaw_idx = (obst_idx == 3)
                reset_seesaw_idx = (reset_seesaw_idx.nonzero(as_tuple=False).flatten())[env_ids]
                self.obst_dof_pos[reset_seesaw_idx[~seesaw_passed]] = self.obstacle.seesaw_dof_pos
                self.obst_dof_pos[reset_seesaw_idx[seesaw_passed]] = -self.obstacle.seesaw_dof_pos
                self.obst_dof_vel[:] = 0.0
            else:
                reset_seesaw_idx = (obst_idx == 3)
                reset_seesaw_idx = (reset_seesaw_idx.nonzero(as_tuple=False).flatten())[env_ids]
                self.obst_dof_pos[reset_seesaw_idx] = self.obstacle.seesaw_dof_pos
                self.obst_dof_vel[:] = 0.0
            dof_states = torch.cat([self.dof_state, self.obst_dof_state], dim=0)
            seesaw_ids = (seesaw_idx.nonzero(as_tuple=False).flatten())[env_ids] + self.num_envs
            env_ids_int32 = torch.cat([env_ids, seesaw_ids]).to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(dof_states),
                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        else:
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environment ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            if self.obstacle.cfg.randomize_start:
                self.root_states[env_ids, :2] += self.root_init_pos[env_ids, :2]
            else:
                # self.root_states[env_ids, :2] += torch.tensor(self.obstacle.cfg.robot_org, device=self.device)
                # self.root_states[env_ids, :3] += self.env_origins[env_ids]
                self.root_states[env_ids, :2] = self.env_goals[env_ids, 0, :2].clone()
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.1, 0.1, (len(env_ids), 2),
                                                                  device=self.device)  # xy position within 1m of the center
            rand_yaw = 0
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range * torch_rand_float(-1, 1, (len(env_ids), 1),
                                                                          device=self.device).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range * torch_rand_float(-1, 1, (len(env_ids), 1),
                                                                                  device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
            if self.obstacle.cfg.randomize_start:
                root_yaw = rand_yaw + self.root_init_ang[env_ids]
            else:
                root_yaw = rand_yaw + self.obstacle.frame_ang[0]
            quat = quat_from_euler_xyz(0 * root_yaw, rand_pitch, root_yaw)
            self.root_states[env_ids, 3:7] = quat[:, :]
            if self.cfg.env.randomize_start_x:
                self.root_states[env_ids, 0] += self.cfg.env.rand_x_range * \
                                                torch_rand_float(-1, 0, (len(env_ids), 1), device=self.device).squeeze(
                                                    1)
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * \
                                                torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(
                                                    1)

        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if self.obstacle:
            if self.cfg.depth.use_camera and self.obstacle.cfg.randomize_border:
                h_r = self.obstacle.cfg.border_height_range
                self.border_root_states[:, 2] = -((h_r[1] - h_r[0]) * torch.rand(self.obstacle.num_envs) + h_r[0])
            root_states = torch.cat([self.root_states, self.obst_root_states, self.border_root_states], dim=0)
            env_ids_int32 = torch.cat([env_ids, env_ids + len(self.root_states),
                                       env_ids + len(self.root_states) + len(self.obst_root_states)],
                                      dim=-1).to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(root_states),
                                                         gymtorch.unwrap_tensor(env_ids_int32),
                                                         len(env_ids_int32))
        else:
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_states),
                                                         gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2),
                                                    device=self.device)  # lin vel x/y
        if self.obstacle:
            root_states = torch.cat([self.root_states, self.obst_root_states, self.border_root_states], dim=0)
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_states))
        else:
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_obstacle_curriculum(self, env_ids):
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        if ((self.success_rate > self.obstacle.curr_threshold) and
                (self.global_counter % (self.max_episode_length * 2) == 0)):
            self.obst_curr_count += 1
            bar_jump_idx = (torch.flatten(self.obstacle_types) == 0)
            seesaw_idx = (torch.flatten(self.obstacle_types) == 3)
            tire_jump_idx = (torch.flatten(self.obstacle_types) == 4)
            self.obst_dof_idx = (bar_jump_idx | seesaw_idx | tire_jump_idx)

            self.bar_jump_bias = min(self.obst_curr_count * self.obstacle.curr_step,
                                     self.obstacle.cfg.bar_jump_max_range[1] - self.obstacle.cfg.bar_jump_max_range[0])
            self.tire_jump_bias = min(self.obst_curr_count * self.obstacle.curr_step,
                                      self.obstacle.cfg.tire_jump_max_range[1] - self.obstacle.cfg.tire_jump_max_range[
                                          0])
            reset_dof_pos = self.reset_dof_pos.clone()
            reset_dof_pos[bar_jump_idx] += self.bar_jump_bias
            reset_dof_pos[tire_jump_idx] += self.tire_jump_bias
            self.target_obst_dof_pos = reset_dof_pos[self.obst_dof_idx].unsqueeze(1)

            height_field_raw = copy.deepcopy(self.obstacle.height_field_raw)
            height_field_raw[self.obstacle.bar_jump_mask] += int(self.bar_jump_bias / self.obstacle.vertical_scale)
            height_field_raw[self.obstacle.tire_jump_mask] += int(self.tire_jump_bias / self.obstacle.vertical_scale)
            self.height_samples = torch.tensor(height_field_raw). \
                view(self.obstacle.tot_rows, self.obstacle.tot_cols).to(self.device)

            env_goals = copy.deepcopy(self.obstacle.env_goals)
            env_goals[self.obstacle.bar_jump_goal_mask] += self.bar_jump_bias
            env_goals[self.obstacle.tire_jump_goal_mask] += self.tire_jump_bias
            self.env_goals = torch.tensor(env_goals, dtype=torch.float, device=self.device)
            self.env_goals = self.env_goals.view(self.env_goals.size(0), -1, self.env_goals.size(-1))
            last_goal_repeat = []
            for k in range(self.obstacle.last_goal_repeat):
                last_goal = self.env_goals[:, -1, :].unsqueeze(1).clone()
                last_goal[:, :, 1] += 0.1 * (k + 1)
                last_goal_repeat.append(last_goal)
            last_goal_repeat = torch.cat(last_goal_repeat, dim=1)
            self.env_goals = torch.cat([self.env_goals, last_goal_repeat], dim=1)

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # (64->128, 13)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # (768->803, 2)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)  # (1216->1315, 3)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)  # (256->256, 6)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)  # (1216->1315, 13)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        if self.obstacle:
            num_obst = self.obstacle.num_envs * self.obstacle.num_obst_per_env
            num_border = self.obstacle.num_envs
            num_obst_links = int(torch.sum(self.num_obst_links).item())
            num_obst_joints = int(torch.sum(self.num_obst_joints).item())
            self.root_states = gymtorch.wrap_tensor(actor_root_state)[:-(num_obst + num_border)]
            self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor)[:-(num_obst_links + num_border)]. \
                view(self.num_envs, -1, 13)
            self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)[:-num_obst_joints]
            self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
            self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
            self.base_quat = self.root_states[:, 3:7]

            self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6)
            self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:-(num_obst_links + num_border)]. \
                view(self.num_envs, -1, 3)

            self.obst_root_states = gymtorch.wrap_tensor(actor_root_state)[-(num_obst + num_border):-num_border]
            self.obst_dof_state = gymtorch.wrap_tensor(dof_state_tensor)[-num_obst_joints:]
            self.obst_dof_pos = self.obst_dof_state.view(num_obst_joints, 1, 2)[..., 0]
            self.obst_dof_vel = self.obst_dof_state.view(num_obst_joints, 1, 2)[..., 1]

            self.border_root_states = gymtorch.wrap_tensor(actor_root_state)[-num_border:]
        else:
            self.root_states = gymtorch.wrap_tensor(actor_root_state)
            self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
            self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
            self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
            self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
            self.base_quat = self.root_states[:, 3:7]

            self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6)
            self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        self.rigid_body_pos = self.rigid_body_states[..., 0:3]
        self.rigid_body_rot = self.rigid_body_states[..., 3:7]
        self.rigid_body_vel = self.rigid_body_states[..., 7:10]
        self.rigid_body_ang_vel = self.rigid_body_states[..., 10:13]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.torques_org = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                       requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques_org = torch.zeros_like(self.torques_org)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions,
                                                                     dtype=torch.float, device=self.device,
                                                                     requires_grad=False) + str_rng[0]
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio -
                                               self.cfg.env.n_auxiliary, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs,
                                              device=self.device, dtype=torch.float)
        self.action_hl_history_buf = None
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 4, device=self.device,
                                       dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        # self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.latent_eps = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.latent_c = torch.zeros(self.num_envs, self.dim_c, dtype=torch.float,
                                    device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.obstacle.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                               requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(
                self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[1],
                                            self.cfg.depth.resized[0]).to(self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.border
        hf_params.transform.p.y = -self.terrain.border
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.flatten(order='C'), hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(
            self.device)

    def _create_obstacle(self):
        self.height_samples = torch.tensor(self.obstacle.height_field_raw).view(self.obstacle.tot_rows,
                                                                                self.obstacle.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.obstacle.x_edge_mask).view(self.obstacle.tot_rows,
                                                                        self.obstacle.tot_cols).to(self.device)

    # def _refresh_obstacle(self):
    #     self.height_samples = torch.tensor(self.obstacle.height_field_raw).view(self.obstacle.tot_rows,
    #                                                                             self.obstacle.tot_cols).to(self.device)
    #     self.x_edge_mask = torch.tensor(self.obstacle.x_edge_mask).view(self.obstacle.tot_rows,
    #                                                                     self.obstacle.tot_cols).to(self.device)
    #     self.env_goals = torch.tensor(self.obstacle.env_goals, dtype=torch.float, device=self.device)
    #     self.env_goals = self.env_goals.view(self.env_goals.size(0), -1, self.env_goals.size(-1))
    #     last_goal_repeat = []
    #     for k in range(self.obstacle.last_goal_repeat):
    #         last_goal = self.env_goals[:, -1, :].unsqueeze(1).clone()
    #         last_goal[:, :, 1] += 0.1 * (k + 1)
    #         last_goal_repeat.append(last_goal)
    #     last_goal_repeat = torch.cat(last_goal_repeat, dim=1)
    #     self.env_goals = torch.cat([self.env_goals, last_goal_repeat], dim=1)

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            camera_position = np.copy(config.position)
            camera_angle = np.random.uniform(config.angle[0], config.angle[1])

            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform,
                                           gymapi.FOLLOW_TRANSFORM)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # load robot
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # load obstacles
        if self.obstacle:
            obstacle_assets = []
            for file in self.obstacle.cfg.files:
                asset_path = file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
                asset_root = os.path.dirname(asset_path)
                asset_file = os.path.basename(asset_path)

                asset_options = gymapi.AssetOptions()
                asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
                asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
                asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
                asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
                asset_options.fix_base_link = True
                asset_options.density = self.cfg.asset.density
                asset_options.angular_damping = self.cfg.asset.angular_damping
                asset_options.linear_damping = self.cfg.asset.linear_damping
                asset_options.max_angular_velocity = 64
                asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
                asset_options.armature = self.cfg.asset.armature
                asset_options.thickness = self.cfg.asset.thickness
                asset_options.disable_gravity = self.cfg.asset.disable_gravity
                asset_options.vhacd_enabled = True

                obstacle_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
                obstacle_assets.append(obstacle_asset)

        asset_path = self.obstacle.cfg.border_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        border_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        for s in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel +
                                self.cfg.init_state.ang_vel)
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.obstacle_handles = []
        self.border_handles = []
        self.obstacle_types = []
        self.cur_obstacle_types = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                              requires_grad=False)

        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.env_handle = env_handle
            pos = self.env_origins[i].clone()
            # pos[:2] += torch.tensor(self.obstacle.cfg.robot_org, device=self.device)
            pos[:2] = self.env_goals[i, 0, :2].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            root_yaw = self.obstacle.frame_ang[0]
            if self.cfg.env.randomize_start_yaw:
                root_yaw += self.cfg.env.rand_yaw_range * np.random.uniform(-1, 1)
            rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., root_yaw)
            start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "go2", i,
                                                 self.cfg.asset.self_collisions, 0)
            self.actor_handle = actor_handle
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            self.attach_camera(i, env_handle, actor_handle)

            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

        # create obstacles
        if self.obstacle:
            print("Creating obstacles...")
            self.obst_stiffness = []
            self.obst_damping = []
            for i in tqdm(range(self.num_envs)):
                env_handle = self.envs[i]
                for j in range(self.obstacle.num_obst_per_env):
                    obstacle_origin = self.obstacle.obstacle_origins[i, j]
                    obstacle_type = self.obstacle.obstacle_types[i, j]
                    obstacle_yaws = self.obstacle.obstacle_yaws[i, j]
                    if obstacle_type == 2:
                        center_point = obstacle_origin + np.array([1.5, 0, 0])
                    elif obstacle_type == 5:
                        center_point = obstacle_origin + np.array([1.0, 0, 0])
                    else:
                        center_point = obstacle_origin
                    offset = torch.tensor(obstacle_origin - center_point, dtype=torch.float32, device=self.device)
                    rotation_matrix = torch.tensor([
                        [math.cos(obstacle_yaws), -math.sin(obstacle_yaws), 0.0],
                        [math.sin(obstacle_yaws), math.cos(obstacle_yaws), 0.0],
                        [0.0, 0.0, 1.0]
                    ], device=self.device)
                    rotated_offset = torch.matmul(rotation_matrix, offset)
                    new_position = torch.tensor(center_point, dtype=torch.float32, device=self.device) + rotated_offset
                    obstacle_ang = np.array([-np.pi / 2, 0, obstacle_yaws])
                    obstacle_quat = rpy2quaternion(obstacle_ang)
                    pose = gymapi.Transform()
                    pose.p = gymapi.Vec3(*new_position)
                    pose.r = gymapi.Quat(*obstacle_quat)

                    # Configure rigid shape properties
                    props = self.gym.get_asset_rigid_shape_properties(obstacle_assets[obstacle_type])
                    for s in range(len(props)):
                        props[s].friction = self.friction_coeffs[i]
                    self.gym.set_asset_rigid_shape_properties(obstacle_assets[obstacle_type], props)

                    obstacle_handle = self.gym.create_actor(env_handle, obstacle_assets[obstacle_type],
                                                            pose, "obstacle", i, 0, 0)
                    self.obstacle_handles.append(obstacle_handle)

                    # Configure DOF properties
                    if obstacle_type in [0, 3, 4]:
                        if obstacle_type == 3:  # seesaw
                            stiffness = 0.0
                            damping = np.random.uniform(1, 10)
                        else:
                            stiffness = 20000.0
                            damping = 1000.0
                        props = self.gym.get_actor_dof_properties(env_handle, obstacle_handle)
                        props["driveMode"] = (gymapi.DOF_MODE_POS,)
                        props["stiffness"] = (stiffness,)
                        props["damping"] = (damping,)
                        self.gym.set_actor_dof_properties(env_handle, obstacle_handle, props)
                        if obstacle_type == 3:  # seesaw
                            stiffness = 0.0
                            damping = 0.0
                        self.obst_stiffness.append(stiffness)
                        self.obst_damping.append(damping)

            for i in tqdm(range(self.num_envs)):
                env_handle = self.envs[i]
                obstacle_origin = self.env_origins[i].clone()
                if self.cfg.depth.use_camera and self.obstacle.cfg.randomize_border:
                    h_r = self.obstacle.cfg.border_height_range
                    obstacle_origin[-1] -= (h_r[1] - h_r[0]) * torch.rand(1)[0] + h_r[0]
                else:
                    obstacle_origin[-1] -= self.obstacle.cfg.border_height
                obstacle_ang = np.array([-np.pi / 2, 0, 0])
                obstacle_quat = rpy2quaternion(obstacle_ang)
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(*obstacle_origin)
                pose.r = gymapi.Quat(*obstacle_quat)

                # Configure rigid shape properties
                props = self.gym.get_asset_rigid_shape_properties(border_asset)
                for s in range(len(props)):
                    props[s].friction = self.friction_coeffs[i]
                self.gym.set_asset_rigid_shape_properties(border_asset, props)

                border_handle = self.gym.create_actor(env_handle, border_asset, pose, "obstacle", i, 0, 0)
                self.border_handles.append(border_handle)

            self.obst_stiffness = torch.tensor(self.obst_stiffness, dtype=torch.float, device=self.device)
            self.obst_damping = torch.tensor(self.obst_damping, dtype=torch.float, device=self.device)
            self.obstacle_types = torch.tensor(self.obstacle.obstacle_types, dtype=torch.long, device=self.device)
            self.cur_obstacle_types = self.obstacle_types[:, 0].clone()
            self.obst_statistics = torch.zeros(len(self.obstacle.proportions), device=self.device)
            self.num_obst_links = torch.zeros(len(self.obstacle.proportions), device=self.device)
            self.num_obst_joints = torch.zeros(len(self.obstacle.proportions), device=self.device)

            bar_jump_idx = (torch.flatten(self.obstacle_types) == 0)
            seesaw_idx = (torch.flatten(self.obstacle_types) == 3)
            tire_jump_idx = (torch.flatten(self.obstacle_types) == 4)
            self.reset_dof_pos = torch.tensor(self.obstacle.obstacle_joint_pos, dtype=torch.float32,
                                              device=self.device).flatten()
            self.obst_dof_idx = (bar_jump_idx | seesaw_idx | tire_jump_idx)
            self.target_obst_dof_pos = self.reset_dof_pos[self.obst_dof_idx].unsqueeze(1)
            for idx in range(len(self.obstacle.proportions)):
                num_obst = torch.sum(self.obstacle_types == idx).item()
                num_obst_links = num_obst * self.obstacle.num_links_per_obst[idx]
                num_obst_joints = num_obst * self.obstacle.num_joints_per_obst[idx]
                self.obst_statistics[idx] = num_obst
                self.num_obst_links[idx] = num_obst_links
                self.num_obst_joints[idx] = num_obst_joints

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        thigh_names = ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)
        calf_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(calf_names):
            self.calf_indices[i] = self.dof_names.index(name)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(self.num_envs, self.cfg.obstacle.num_goals + self.cfg.env.num_future_goal_obs,
                                         3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[
                                :]  # (64, 10, 3)
            self.cur_goals = self._gather_cur_goals()  # (64, 3)
            self.next_goals = self._gather_cur_goals(future=1)
        elif self.cfg.terrain.mesh_type == "obstacle":
            self.custom_origins = True
            self.env_origins = torch.tensor(self.obstacle.env_origins, dtype=torch.float, device=self.device)
            self.env_class = torch.tensor(self.obstacle.obstacle_types, dtype=torch.float, device=self.device)
            self.cur_goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.env_goals = torch.tensor(self.obstacle.env_goals, dtype=torch.float, device=self.device)
            self.env_goals = self.env_goals.view(self.env_goals.size(0), -1, self.env_goals.size(-1))
            last_goal_repeat = []
            for k in range(self.obstacle.last_goal_repeat):
                last_goal = self.env_goals[:, -1, :].unsqueeze(1).clone()
                last_goal[:, :, 1] += 0.1 * (k + 1)
                last_goal_repeat.append(last_goal)
            last_goal_repeat = torch.cat(last_goal_repeat, dim=1)
            self.env_goals = torch.cat([self.env_goals, last_goal_repeat], dim=1)
            if self.obstacle.cfg.randomize_start:
                self.cur_obst_idx = torch.randint_like(self.cur_goal_idx, 0, self.obstacle.env_goals.shape[1])
                self.cur_goal_idx = self.cur_obst_idx * self.obstacle.env_goals.shape[2]
                self.root_init_pos = self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]). \
                                                           expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)
                self.obst_angs = torch.tensor(self.obstacle.frame_ang, device=self.device, dtype=torch.float32). \
                    expand(self.env_goals.shape[0], -1)
                self.root_init_ang = self.obst_angs.gather(1, (self.cur_obst_idx[:, None])).squeeze(1)
            self.cur_goals = self._gather_cur_goals()  # (64, 3)
            self.next_goals = self._gather_cur_goals(future=1)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        reward_norm_factor = 1  # np.sum(list(self.reward_scales.values()))
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_height_samples(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if self.obstacle:
            if not self.obstacle.cfg.measure_heights:
                return
        else:
            if not self.terrain.cfg.measure_heights:
                return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0.96, 0.49, 0.13))
        i = self.lookat_id
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_goals(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))
        sphere_geom_cur = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))
        sphere_geom_reached = gymutil.WireframeSphereGeometry(self.cfg.env.next_goal_threshold, 32, 32, None,
                                                              color=(0, 1, 0))
        if self.obstacle:
            goals = self.env_goals[self.lookat_id].reshape(-1, 3)
        else:
            goals = self.terrain_goals[
                self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]].cpu().numpy()
        for i, goal in enumerate(goals):
            if self.obstacle:
                pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal[2]), r=None)
            else:
                goal_xy = goal[:2] + self.terrain.cfg.border_size
                pts = (goal_xy / self.terrain.cfg.horizontal_scale).astype(int)
                goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
                pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
            if i == self.cur_goal_idx[self.lookat_id].cpu().item():
                gymutil.draw_lines(sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                if self.reached_goal_ids[self.lookat_id]:
                    gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            else:
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
            for i in range(5):
                norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.1 * (i + 3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)

            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
            for i in range(5):
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.2 * (i + 3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def _draw_feet(self):
        if hasattr(self, 'feet_at_edge'):
            non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
            edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0, 0))

            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            for i in range(4):
                pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1],
                                                    feet_pos[self.lookat_id, i, 2]), r=None)
                if self.feet_at_edge[self.lookat_id, i]:
                    gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[i], pose)
                else:
                    gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[i], pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        if self.obstacle:
            y = torch.tensor(self.cfg.obstacle.measured_points_y, device=self.device, requires_grad=False)
            x = torch.tensor(self.cfg.obstacle.measured_points_x, device=self.device, requires_grad=False)
        else:
            y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
            x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise,
                                      self.cfg.terrain.measure_horizontal_noise, (self.num_height_points, 2),
                                      device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise,
                                        self.cfg.terrain.measure_horizontal_noise, (self.num_height_points, 2),
                                        device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def get_foot_contacts(self):
        foot_contacts_bool = self.contact_forces[:, self.feet_indices, 2] > 10
        if self.cfg.env.include_foot_contacts:
            return foot_contacts_bool
        else:
            return torch.zeros_like(foot_contacts_bool).to(self.device)

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]).unsqueeze(1)

        if self.obstacle:
            border_size = self.obstacle.cfg.border_size
            horizontal_scale = self.obstacle.cfg.horizontal_scale
            vertical_scale = self.obstacle.cfg.vertical_scale
        else:
            border_size = self.terrain.cfg.border_size
            horizontal_scale = self.terrain.cfg.horizontal_scale
            vertical_scale = self.terrain.cfg.vertical_scale

        points += border_size
        points = (points / horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * vertical_scale

    def _get_heights_points(self, coords, env_ids=None):
        if env_ids:
            points = coords[env_ids]
        else:
            points = coords

        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    ################## parkour rewards ##################

    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:9]
        rew = (torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0] * 0 + self.cfg.rewards.target_lin_vel) / (
                self.commands[:, 0] * 0 + self.cfg.rewards.target_lin_vel + 1e-5))
        rew[self.cur_obstacle_types == 0] = \
            (torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0] * 0 + 2.5) / (
                    self.commands[:, 0] * 0 + 2.5 + 1e-5))[self.cur_obstacle_types == 0]
        rew[self.cur_obstacle_types == 4] = \
            (torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0] * 0 + 2.5) / (
                    self.commands[:, 0] * 0 + 2.5 + 1e-5))[self.cur_obstacle_types == 4]
        return rew

    def _reward_tracking_yaw(self):
        # Normalize to [-pi, pi]
        delta_yaw = ((self.target_yaw - self.yaw) + torch.pi) % (2 * torch.pi) - torch.pi
        rew = torch.exp(-torch.abs(delta_yaw))
        return rew

    # def _reward_tracking_goal_vel(self):
    #     norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
    #     target_vec_norm = self.target_pos_rel / (norm + 1e-5)
    #     cur_vel = self.root_states[:, 7:9]
    #
    #     target_vel_0 = torch.zeros_like(self.commands[:, 0]) + 1.0  # poles
    #     target_vel_1 = torch.zeros_like(self.commands[:, 0]) + 1.5  # others
    #     target_vel_2 = torch.zeros_like(self.commands[:, 0]) + 2.5  # seesaw, frame
    #     projected_vel = torch.sum(target_vec_norm * cur_vel, dim=-1)
    #
    #     rew = torch.exp(-torch.square(target_vel_1 - projected_vel) / self.cfg.rewards.tracking_sigma)
    #     rew[self.cur_obstacle_types == 2] = \
    #         (torch.exp(-torch.square(target_vel_0 - projected_vel) / self.cfg.rewards.tracking_sigma))[self.cur_obstacle_types == 2]
    #     rew[self.cur_obstacle_types == 0] = \
    #         (torch.exp(-torch.square(target_vel_2 - projected_vel) / self.cfg.rewards.tracking_sigma))[self.cur_obstacle_types == 0]
    #     rew[self.cur_obstacle_types == 4] = \
    #         (torch.exp(-torch.square(target_vel_2 - projected_vel) / self.cfg.rewards.tracking_sigma))[self.cur_obstacle_types == 4]
    #     return rew
    #
    # def _reward_tracking_yaw(self):
    #     # Normalize to [-pi, pi]
    #     delta_yaw = ((self.target_yaw - self.yaw) + torch.pi) % (2 * torch.pi) - torch.pi
    #     rew = torch.exp(-torch.square(delta_yaw) / self.cfg.rewards.tracking_sigma)
    #     return rew

    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        # rew[self.env_class != 17] *= 0.5
        return rew

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # rew[self.env_class != 17] = 0.
        return rew

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_action_hl_rate(self):
        if self.action_hl_history_buf is None:
            return 0
        else:
            return torch.norm(self.action_hl_history_buf[:, -2, :] - self.action_hl_history_buf[:, -1, :], dim=1)

    def _reward_latent_c_rate(self):
        if self.action_hl_history_buf is None:
            return 0
        else:
            latent_diff1 = torch.abs(self.action_hl_history_buf[:, -3, 0] - self.action_hl_history_buf[:, -1, 0])
            latent_diff2 = torch.abs(self.action_hl_history_buf[:, -2, 0] - self.action_hl_history_buf[:, -1, 0])
            return 0.5 * (latent_diff1 + latent_diff2)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques_org - self.last_torques_org), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques_org), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques_org) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]),
                         dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                        4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_edge(self):
        if self.obstacle:
            border_size = self.obstacle.cfg.border_size
            horizontal_scale = self.cfg.obstacle.horizontal_scale
        else:
            border_size = self.terrain.cfg.border_size
            horizontal_scale = self.cfg.terrain.horizontal_scale
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices,
                        :2] + border_size) / horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

        self.feet_at_edge = self.contact_filt & feet_at_edge
        # rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        rew = torch.sum(self.feet_at_edge, dim=-1)
        return rew

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_reach_goal(self):
        return self.reached_goal_ids

    def _reward_every_step(self):
        return torch.ones(self.num_envs, device=self.device, dtype=torch.float32)


@torch.jit.script
def compute_flat_key_pos(root_states, key_body_pos):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = calc_heading_quat_inv(root_rot)
    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
                                           local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0],
                                            local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    return flat_local_key_pos
