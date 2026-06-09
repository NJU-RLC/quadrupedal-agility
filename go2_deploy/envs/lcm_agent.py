import time

import lcm
import numpy as np
import torch

from go2_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt


def class_to_dict(obj):
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        value = getattr(obj, key)
        result[key] = [class_to_dict(item) for item in value] if isinstance(value, list) else class_to_dict(value)
    return result


class LCMAgent:
    """Hardware-facing agent that builds observations and publishes PD targets."""

    JOINT_NAMES = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]

    def __init__(self, cfg, state_estimator, command_profile, device="cuda", lcm_url="udpm://239.255.76.67:7667?ttl=255"):
        self.cfg = class_to_dict(cfg) if not isinstance(cfg, dict) else cfg
        self.se = state_estimator
        self.command_profile = command_profile
        self.device = torch.device(device)
        self.lc = lcm.LCM(lcm_url)

        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        self.timestep = 0
        self.num_envs = 1
        self.num_obs = self.cfg["env"]["num_obs"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.num_commands = self.cfg["commands"]["num_commands"]
        self.num_prop = self.cfg["env"]["num_prop"]
        self.obs_scales = self.cfg.get("obs_scales", self.cfg["normalization"]["obs_scales"])

        self.default_dof_pos = np.array([self.cfg["init_state"]["default_joint_angles"][name] for name in self.JOINT_NAMES])
        scale = self.cfg["init_state"]
        dof_scale = np.array(
            [
                scale.get("default_hip_scales", 1.0),
                scale.get("default_thigh_scales", 1.0),
                scale.get("default_calf_scales", 1.0),
            ]
            * 4
        )
        self.default_dof_pos *= dof_scale
        self.start_pos = np.array([0.0, 0.9, -1.5] * 4)

        stiffness = self.cfg["control"]["stiffness"]
        damping = self.cfg["control"]["damping"]
        self.p_gains = np.array([self._match_gain(stiffness, joint) for joint in self.JOINT_NAMES])
        self.d_gains = np.array([self._match_gain(damping, joint) for joint in self.JOINT_NAMES])

        self.commands = np.zeros((1, self.num_commands))
        self.last_actions = np.zeros(12)
        self.joint_pos_target = np.zeros(12)
        self.joint_vel_target = np.zeros(12)
        self.torques = np.zeros(12)
        self.joint_idxs = self.se.joint_idxs
        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        self.body_angular_vel = np.zeros(3)
        self.contact_state = np.zeros(4)
        self.latent_c = np.zeros(len(self.cfg["env"]["mocap_category"]))
        self.latent_eps = 0
        self.is_currently_probing = False
        self.next_time = None

    @staticmethod
    def _match_gain(gains, joint_name):
        for dof_name, value in gains.items():
            if dof_name in joint_name or dof_name == "joint":
                return value
        return 0.0

    def set_probing(self, is_currently_probing):
        self.is_currently_probing = is_currently_probing

    def get_obs(self):
        cmds, _ = self.command_profile.get_command(
            self.timestep * self.dt,
            mode_number=self.se.mode_number,
            probe=self.is_currently_probing,
        )
        self.commands[:, :] = cmds[: self.num_commands]

        rpy = self.se.get_rpy()
        self.dof_pos = self.se.get_dof_pos()
        self.dof_vel = self.se.get_dof_vel()
        self.body_angular_vel = self.se.get_body_angular_vel()
        self.latent_c = self.se.get_latent()
        self.latent_eps = self.se.get_latent_eps()
        self.contact_state = self.se.get_contact_state()

        dof_pos_obs = (self.dof_pos - self.default_dof_pos).reshape(1, -1) * self.obs_scales["dof_pos"]
        dof_vel_obs = self.dof_vel.reshape(1, -1) * self.obs_scales["dof_vel"]
        imu_obs = np.array(rpy[:2]).reshape(1, -1)
        imu_yaw = np.array(rpy[2]).reshape(1, -1)
        base_ang_vel_obs = np.array(self.body_angular_vel).reshape(1, -1) * self.obs_scales["ang_vel"]
        action_obs = np.array(self.last_actions).reshape(1, -1)
        contact_obs = np.array(self.contact_state).reshape(1, -1)
        key_pos_obs = np.zeros(12).reshape(1, -1)

        obs_main = np.concatenate((imu_obs, base_ang_vel_obs, dof_pos_obs, dof_vel_obs, action_obs, contact_obs, key_pos_obs), axis=1)
        obs_command = np.concatenate(
            (
                self.commands.reshape(1, -1),
                np.array(self.latent_eps).reshape(1, -1),
                self.latent_c.reshape(1, -1),
            ),
            axis=1,
        )

        obs_vis = self.se.get_depth_image() if self.se.use_cameras else None
        return {
            "obs_main": torch.from_numpy(obs_main).to(self.device).half(),
            "obs_yaw": torch.from_numpy(imu_yaw).to(self.device).half(),
            "obs_command": torch.tensor(obs_command, device=self.device).half(),
            "obs_vis": obs_vis,
        }

    def publish_action(self, action_numpy, hard_reset=False, start_sig=False):
        msg = pd_tau_targets_lcmt()
        self.joint_pos_target = (action_numpy[0][:12] * self.cfg["control"]["action_scale"]).flatten()
        self.joint_pos_target[[0, 3, 6, 9]] *= self.cfg["control"]["hip_scale_reduction"]
        self.joint_pos_target += self.start_pos if start_sig else self.default_dof_pos

        msg.q_des = self.joint_pos_target[self.joint_idxs]
        msg.qd_des = np.zeros(12)
        msg.kp = np.array([40.0] * 12) if start_sig else self.p_gains
        msg.kd = np.array([1.0] * 12) if start_sig else self.d_gains
        msg.tau_ff = np.zeros(12)
        msg.se_contactState = np.zeros(4)
        msg.timestamp_us = int(time.time() * 1e6)
        msg.id = -1 if hard_reset else 0

        self.joint_vel_target = np.zeros(12)
        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (self.joint_vel_target - self.dof_vel) * self.d_gains
        self.lc.publish("pd_plustau_targets", msg.encode())

    def reset(self):
        self.last_actions = np.zeros(12)
        self.timestep = 0
        self.next_time = None
        self.time_frequent = time.perf_counter()
        return self.get_obs()

    def step(self, actions, hard_reset=False, start_sig=False):
        action_numpy = actions.to("cpu").detach().numpy()
        clip_actions = self.cfg["normalization"]["clip_actions"]
        action_scale = self.cfg["control"]["action_scale"]
        clipped = np.clip(action_numpy[0, :], -clip_actions / action_scale, clip_actions / action_scale)
        self.last_actions = action_numpy

        self.publish_action(action_numpy, hard_reset=hard_reset, start_sig=start_sig)

        if self.next_time is None:
            self.next_time = time.perf_counter() + self.dt
        current_time = time.perf_counter()
        if self.next_time > current_time:
            time.sleep(self.next_time - current_time)
        self.next_time += self.dt
        if self.next_time < current_time:
            self.next_time = current_time + self.dt

        if self.timestep % 50 == 0:
            elapsed = max(time.perf_counter() - self.time_frequent, 1e-6)
            camera_msg = f"Camera FPS: {1 / self.se.get_depth_time():.1f}" if self.se.use_cameras and self.se.get_depth_time() else "Camera closed"
            print(f"Control FPS: {50 / elapsed:.1f} Hz | {camera_msg}")
            self.time_frequent = time.perf_counter()

        obs = self.get_obs()
        info = {
            "joint_pos": self.dof_pos[np.newaxis, :],
            "joint_vel": self.dof_vel[np.newaxis, :],
            "joint_pos_target": self.joint_pos_target[np.newaxis, :],
            "joint_vel_target": self.joint_vel_target[np.newaxis, :],
            "body_angular_vel": self.body_angular_vel[np.newaxis, :],
            "body_linear_vel_cmd": self.commands[:, 0:2],
            "body_angular_vel_cmd": self.commands[:, 2],
            "action_clip": clipped,
            "latent_c": self.latent_c,
            "latent_eps": self.latent_eps,
            "privileged_obs": None,
            "tau_est": self.se.get_tau_est(),
        }
        if not start_sig:
            self.timestep += 1
        return obs, None, None, info
