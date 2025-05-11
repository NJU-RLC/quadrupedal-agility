import os
import glob
import json
import logging

import torch
import numpy as np
from pybullet_utils import transformations
from torch import Tensor

from rsl_rl.utils import utils
from rsl_rl.datasets import pose3d
from rsl_rl.datasets import motion_util

from legged_gym.utils.torch_jit_utils import *


class MotionLoader:
    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    TAR_TOE_POS_LOCAL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12
    TAR_TOE_VEL_LOCAL_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    TAR_TOE_VEL_LOCAL_START_IDX = JOINT_VEL_END_IDX
    TAR_TOE_VEL_LOCAL_END_IDX = TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE

    def __init__(
            self,
            device,
            time_between_frames,
            mocap_state_init=False,
            motion_files_lb=None,
            motion_files_ulb=None,
            mocap_category=None,
            num_preload_transitions=1000000,
            compute_flat_key_pos=None,
            default_dof_pos=None,
            obs_scales=None,
            num_disc_obs=44,
            disc_obs_len=2,
            obs_disc_weight_step=0.1,
            frame_duration_scale=1.0
    ):
        """Expert dataset provides observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.mocap_state_init = mocap_state_init
        self.mocap_category = mocap_category
        self.num_disc_obs = num_disc_obs
        self.disc_obs_len = disc_obs_len
        self.obs_disc_weight_step = obs_disc_weight_step

        self.motion_files_lb = motion_files_lb
        self.motion_files_ulb = motion_files_ulb
        self.num_preload_transitions = num_preload_transitions
        self.compute_flat_key_pos = compute_flat_key_pos
        self.default_dof_pos = default_dof_pos
        self.obs_scales = obs_scales
        self.frame_duration_scale = frame_duration_scale

        self.pre_load_data()

    def pre_load_data(self):
        # labeled data
        self.mocap_trajectory_lb = []
        self.mocap_trajectory_full_lb = []
        self.mocap_label = []
        self.mocap_idxs_lb = []
        self.mocap_lens_lb = []
        self.mocap_weights_lb = []
        self.mocap_frame_durations_lb = []
        self.mocap_num_frames_lb = []

        # unlabeled data
        self.mocap_trajectory_ulb = []
        self.mocap_trajectory_full_ulb = []
        self.mocap_idxs_ulb = []
        self.mocap_lens_ulb = []
        self.mocap_weights_ulb = []
        self.mocap_frame_durations_ulb = []
        self.mocap_num_frames_ulb = []

        for i, motion_file in enumerate(self.motion_files_lb):
            label = None
            motion_file_name = motion_file.split('/')
            for idx, cate in enumerate(self.mocap_category):
                if cate in motion_file_name[-1]:
                    label = idx
            if label is None:
                raise ValueError('Unsupported mocap category {}.'.format(motion_file))
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                motion_data = self.reorder(motion_data)

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MotionLoader.get_root_rot(motion_data[f_i])
                    root_rot = pose3d.QuaternionNormalize(root_rot)
                    root_rot = motion_util.standardize_quaternion(root_rot)
                    motion_data[f_i, MotionLoader.ROOT_ROT_START_IDX: MotionLoader.ROOT_ROT_END_IDX] = root_rot

                self.mocap_trajectory_lb.append(torch.tensor(
                    motion_data[:, MotionLoader.ROOT_ROT_END_IDX:MotionLoader.JOINT_VEL_END_IDX],
                    dtype=torch.float32, device=self.device))
                self.mocap_trajectory_full_lb.append(torch.tensor(
                    motion_data[:, :MotionLoader.JOINT_VEL_END_IDX],
                    dtype=torch.float32, device=self.device))
                self.mocap_idxs_lb.append(i)
                self.mocap_weights_lb.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"]) * self.frame_duration_scale
                self.mocap_frame_durations_lb.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.mocap_lens_lb.append(traj_len)
                self.mocap_num_frames_lb.append(float(motion_data.shape[0]))
                self.mocap_label.append(label)

        # Trajectory weights are used to sample some trajectories more than others.
        self.mocap_weights_lb = np.array(self.mocap_weights_lb) / np.sum(self.mocap_weights_lb)
        self.mocap_frame_durations_lb = np.array(self.mocap_frame_durations_lb)
        self.mocap_lens_lb = np.array(self.mocap_lens_lb)
        self.mocap_num_frames_lb = np.array(self.mocap_num_frames_lb)
        self.mocap_label = np.array(self.mocap_label)

        for i, motion_file in enumerate(self.motion_files_ulb):
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                motion_data = self.reorder(motion_data)

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MotionLoader.get_root_rot(motion_data[f_i])
                    root_rot = pose3d.QuaternionNormalize(root_rot)
                    root_rot = motion_util.standardize_quaternion(root_rot)
                    motion_data[f_i, MotionLoader.ROOT_ROT_START_IDX: MotionLoader.ROOT_ROT_END_IDX] = root_rot

                self.mocap_trajectory_ulb.append(torch.tensor(
                    motion_data[:, MotionLoader.ROOT_ROT_END_IDX:MotionLoader.JOINT_VEL_END_IDX],
                    dtype=torch.float32, device=self.device))
                self.mocap_trajectory_full_ulb.append(torch.tensor(
                    motion_data[:, :MotionLoader.JOINT_VEL_END_IDX],
                    dtype=torch.float32, device=self.device))
                self.mocap_idxs_ulb.append(i)
                self.mocap_weights_ulb.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"]) * self.frame_duration_scale
                self.mocap_frame_durations_ulb.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.mocap_lens_ulb.append(traj_len)
                self.mocap_num_frames_ulb.append(float(motion_data.shape[0]))

        self.mocap_weights_ulb = np.array([1.0])
        self.mocap_frame_durations_ulb = np.array([self.mocap_frame_durations_ulb[0]])
        self.mocap_lens_ulb = np.array([np.sum(self.mocap_lens_ulb)])
        self.mocap_num_frames_ulb = np.array([np.sum(self.mocap_num_frames_ulb)])
        self.mocap_trajectory_ulb = [torch.cat(self.mocap_trajectory_ulb, dim=0)]
        self.mocap_trajectory_full_ulb = [torch.cat(self.mocap_trajectory_full_ulb, dim=0)]
        self.mocap_idxs_ulb = [0]

        self.preloaded_s_lb = None
        self.preloaded_s_next_lb = None
        self.preloaded_label = None
        self.preloaded_s_ulb = None
        self.preloaded_s_next_ulb = None
        if not self.mocap_state_init:
            traj_idxs = self.weighted_traj_idx_sample_batch(self.num_preload_transitions, labeled=True)
            times = self.traj_time_sample_batch(traj_idxs, labeled=True)
            preloaded_s_lb = []
            for i in range(self.disc_obs_len):
                preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times, labeled=True)
                root_state = torch.cat([preloaded_s[:, :7], preloaded_s[:, 31:34], preloaded_s[:, 34:37]], dim=-1)
                root_h = root_state[:, 2:3]
                dof_pos = preloaded_s[:, 7:19]
                dof_vel = preloaded_s[:, 37:49]
                key_body_pos = preloaded_s[:, 19:31].reshape(-1, 4, 3)

                base_quat = root_state[:, 3:7]
                base_lin_vel = quat_rotate_inverse(base_quat, root_state[:, 7:10])
                base_ang_vel = quat_rotate_inverse(base_quat, root_state[:, 10:13])
                roll, pitch, yaw = euler_from_quaternion(base_quat)
                imu_obs = torch.stack((roll, pitch), dim=1)
                flat_local_key_pos = self.compute_flat_key_pos(root_state, key_body_pos)
                foot_contact = (key_body_pos[:, :, -1] < 0.025).to(torch.float32)
                preloaded_s_lb.append(torch.cat([imu_obs, root_h, base_lin_vel * self.obs_scales.lin_vel_dist,
                                                 base_ang_vel * self.obs_scales.ang_vel_dist,
                                                 (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                                 dof_vel * self.obs_scales.dof_vel,
                                                 flat_local_key_pos * self.obs_scales.key_pos,
                                                 foot_contact * self.obs_scales.foot_contact], dim=-1))
                times = times + self.time_between_frames

            self.preloaded_s_lb = torch.cat(preloaded_s_lb, dim=-1)
            self.preloaded_label = torch.tensor([self.mocap_label[idx] for idx in traj_idxs], device=self.device)

            traj_idxs = self.weighted_traj_idx_sample_batch(self.num_preload_transitions, labeled=False)
            times = self.traj_time_sample_batch(traj_idxs, labeled=False)
            preloaded_s_ulb = []
            for i in range(self.disc_obs_len):
                preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times, labeled=False)
                root_state = torch.cat([preloaded_s[:, :7], preloaded_s[:, 31:34], preloaded_s[:, 34:37]], dim=-1)
                root_h = root_state[:, 2:3]
                dof_pos = preloaded_s[:, 7:19]
                dof_vel = preloaded_s[:, 37:49]
                key_body_pos = preloaded_s[:, 19:31].reshape(-1, 4, 3)

                base_quat = root_state[:, 3:7]
                base_lin_vel = quat_rotate_inverse(base_quat, root_state[:, 7:10])
                base_ang_vel = quat_rotate_inverse(base_quat, root_state[:, 10:13])
                roll, pitch, yaw = euler_from_quaternion(base_quat)
                imu_obs = torch.stack((roll, pitch), dim=1)
                flat_local_key_pos = self.compute_flat_key_pos(root_state, key_body_pos)
                foot_contact = (key_body_pos[:, :, -1] < 0.025).to(torch.float32)
                preloaded_s_ulb.append(torch.cat([imu_obs, root_h, base_lin_vel * self.obs_scales.lin_vel_dist,
                                                  base_ang_vel * self.obs_scales.ang_vel_dist,
                                                  (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                                  dof_vel * self.obs_scales.dof_vel,
                                                  flat_local_key_pos * self.obs_scales.key_pos,
                                                  foot_contact * self.obs_scales.foot_contact], dim=-1))
                times = times + self.time_between_frames

            self.preloaded_s_ulb = torch.cat(preloaded_s_ulb, dim=-1)

    @staticmethod
    def reorder(motion_data):
        """Convert from PyBullet ordering to Isaac ordering.

        Rearranges leg and joint order from PyBullet [FR, FL, RR, RL] to
        IsaacGym order [FL, FR, RL, RR].
        """
        root_pos = MotionLoader.get_root_pos_batch(motion_data)
        root_rot = MotionLoader.get_root_rot_batch(motion_data)

        jp_fr, jp_fl, jp_rr, jp_rl = np.split(
            MotionLoader.get_joint_pose_batch(motion_data), 4, axis=1)

        jp_fr[:, 0] = -jp_fr[:, 0]
        jp_fl[:, 0] = -jp_fl[:, 0]
        jp_rr[:, 0] = -jp_rr[:, 0]
        jp_rl[:, 0] = -jp_rl[:, 0]
        joint_pos = np.hstack([jp_fl, jp_fr, jp_rl, jp_rr])

        fp_fr, fp_fl, fp_rr, fp_rl = np.split(
            MotionLoader.get_tar_toe_pos_local_batch(motion_data), 4, axis=1)

        # Move the foot to the ground
        foot_fl_h_min = np.min(fp_fl[:, -1])
        foot_fr_h_min = np.min(fp_fr[:, -1])
        foot_rl_h_min = np.min(fp_rl[:, -1])
        foot_rr_h_min = np.min(fp_rr[:, -1])
        root_pos[:, -1] -= np.mean([foot_fl_h_min, foot_fr_h_min, foot_rl_h_min, foot_rr_h_min])
        fp_fl[:, -1] -= foot_fl_h_min
        fp_fr[:, -1] -= foot_fr_h_min
        fp_rl[:, -1] -= foot_rl_h_min
        fp_rr[:, -1] -= foot_rr_h_min

        foot_pos = np.hstack([fp_fl, fp_fr, fp_rl, fp_rr])

        lin_vel = MotionLoader.get_linear_vel_batch(motion_data)
        ang_vel = MotionLoader.get_angular_vel_batch(motion_data)

        jv_fr, jv_fl, jv_rr, jv_rl = np.split(
            MotionLoader.get_joint_vel_batch(motion_data), 4, axis=1)

        jv_fr[:, 0] = -jv_fr[:, 0]
        jv_fl[:, 0] = -jv_fl[:, 0]
        jv_rr[:, 0] = -jv_rr[:, 0]
        jv_rl[:, 0] = -jv_rl[:, 0]
        joint_vel = np.hstack([jv_fl, jv_fr, jv_rl, jv_rr])

        fv_fr, fv_fl, fv_rr, fv_rl = np.split(
            MotionLoader.get_tar_toe_vel_local_batch(motion_data), 4, axis=1)
        foot_vel = np.hstack([fv_fl, fv_fr, fv_rl, fv_rr])

        return np.hstack([root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel, joint_vel, foot_vel])

    def weighted_traj_idx_sample(self, labeled=False):
        """Get traj idx via weighted sampling."""
        if labeled:
            return np.random.choice(self.mocap_idxs_lb, p=self.mocap_weights_lb)
        else:
            return np.random.choice(self.mocap_idxs_ulb, p=self.mocap_weights_ulb)

    def weighted_traj_idx_sample_batch(self, size, labeled=False, target_type=None):
        """Batch sample traj idxs."""
        if labeled:
            if target_type is None:
                return np.random.choice(self.mocap_idxs_lb, size=size, p=self.mocap_weights_lb, replace=True)
            else:
                target_label = self.mocap_label == target_type
                return np.random.choice(np.array(self.mocap_idxs_lb)[target_label], size=size,
                                        p=self.mocap_weights_lb[target_label] / np.sum(
                                            self.mocap_weights_lb[target_label]), replace=True)
        else:
            return np.random.choice(self.mocap_idxs_ulb, size=size, p=self.mocap_weights_ulb, replace=True)

    def traj_time_sample(self, traj_idx, labeled=False):
        """Sample random time for traj."""
        if labeled:
            subst = self.time_between_frames + self.mocap_frame_durations_lb[traj_idx]
            return max(1e-7, ((self.mocap_lens_lb[traj_idx] - subst) * np.random.uniform()))
        else:
            subst = self.time_between_frames + self.mocap_frame_durations_ulb[traj_idx]
            return max(1e-7, ((self.mocap_lens_ulb[traj_idx] - subst) * np.random.uniform()))

    def traj_time_sample_batch(self, traj_idxs, labeled=False):
        """Sample random time for multiple trajectories."""
        if labeled:
            subst = self.time_between_frames * self.disc_obs_len + self.mocap_frame_durations_lb[traj_idxs]
            time_samples = (self.mocap_lens_lb[traj_idxs] - subst) * np.random.uniform(size=len(traj_idxs))
        else:
            subst = self.time_between_frames * self.disc_obs_len + self.mocap_frame_durations_ulb[traj_idxs]
            time_samples = (self.mocap_lens_ulb[traj_idxs] - subst) * np.random.uniform(size=len(traj_idxs))
        return np.maximum(np.zeros_like(time_samples) + 1e-7, time_samples)

    @staticmethod
    def interp(val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx, labeled=False):
        """Returns trajectory of observations."""
        if labeled:
            return self.mocap_trajectory_full_lb[traj_idx]
        else:
            return self.mocap_trajectory_full_ulb[traj_idx]

    def get_frame_at_time(self, traj_idx, time, labeled=False):
        """Returns frame for the given trajectory at the specified time."""
        if labeled:
            p = float(time) / self.mocap_lens_lb[traj_idx]
            n = self.mocap_trajectory_lb[traj_idx].shape[0]
            idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
            frame_start = self.mocap_trajectory_lb[traj_idx][idx_low]
            frame_end = self.mocap_trajectory_lb[traj_idx][idx_high]
        else:
            p = float(time) / self.mocap_lens_ulb[traj_idx]
            n = self.mocap_trajectory_ulb[traj_idx].shape[0]
            idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
            frame_start = self.mocap_trajectory_ulb[traj_idx][idx_low]
            frame_end = self.mocap_trajectory_ulb[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.interp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times, labeled=False):
        """Returns frame for the given trajectory at the specified time."""
        if labeled:
            p = times / self.mocap_lens_lb[traj_idxs]
            n = self.mocap_num_frames_lb[traj_idxs]
        else:
            p = times / self.mocap_lens_ulb[traj_idxs]
            n = self.mocap_num_frames_ulb[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(int), np.ceil(p * n).astype(int)
        all_frame_starts = torch.zeros(len(traj_idxs), self.num_disc_obs, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.num_disc_obs, device=self.device)
        for traj_idx in set(traj_idxs):
            if labeled:
                trajectory = self.mocap_trajectory_lb[traj_idx]
            else:
                trajectory = self.mocap_trajectory_ulb[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.interp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time, labeled=False):
        """Returns full frame for the given trajectory at the specified time."""
        if labeled:
            p = float(time) / self.mocap_lens_lb[traj_idx]
            n = self.mocap_trajectory_full_lb[traj_idx].shape[0]
            idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
            frame_start = self.mocap_trajectory_full_lb[traj_idx][idx_low]
            frame_end = self.mocap_trajectory_full_lb[traj_idx][idx_high]
        else:
            p = float(time) / self.mocap_lens_lb[traj_idx]
            n = self.mocap_trajectory_full_ulb[traj_idx].shape[0]
            idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
            frame_start = self.mocap_trajectory_full_ulb[traj_idx][idx_low]
            frame_end = self.mocap_trajectory_full_ulb[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times, labeled=False):
        if labeled:
            p = times / self.mocap_lens_lb[traj_idxs]
            n = self.mocap_num_frames_lb[traj_idxs]
        else:
            p = times / self.mocap_lens_ulb[traj_idxs]
            n = self.mocap_num_frames_ulb[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(int), np.ceil(p * n).astype(int)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), MotionLoader.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), MotionLoader.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), MotionLoader.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), MotionLoader.ROT_SIZE, device=self.device)
        all_frame_traj_starts = torch.zeros(len(traj_idxs),
                                            MotionLoader.JOINT_VEL_END_IDX - MotionLoader.JOINT_POSE_START_IDX,
                                            device=self.device)
        all_frame_traj_ends = torch.zeros(len(traj_idxs),
                                          MotionLoader.JOINT_VEL_END_IDX - MotionLoader.JOINT_POSE_START_IDX,
                                          device=self.device)
        for traj_idx in set(traj_idxs):
            if labeled:
                trajectory = self.mocap_trajectory_full_lb[traj_idx]
            else:
                trajectory = self.mocap_trajectory_full_ulb[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = MotionLoader.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = MotionLoader.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = MotionLoader.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = MotionLoader.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_traj_starts[traj_mask] = trajectory[idx_low[traj_mask]][:,
                                               MotionLoader.JOINT_POSE_START_IDX:MotionLoader.JOINT_VEL_END_IDX]
            all_frame_traj_ends[traj_mask] = trajectory[idx_high[traj_mask]][:,
                                             MotionLoader.JOINT_POSE_START_IDX:MotionLoader.JOINT_VEL_END_IDX]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.interp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        traj_blend = self.interp(all_frame_traj_starts, all_frame_traj_ends, blend)
        return torch.cat([pos_blend, rot_blend, traj_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames, latent_c_idx=None):
        # if self.mocap_state_init:
        #     idxs = np.random.choice(
        #         self.preloaded_s.shape[0], size=num_frames)
        #     return self.preloaded_s[idxs]
        # else:
        traj_idxs = np.zeros(num_frames, dtype=np.int64)
        if latent_c_idx is not None:
            for i in range(len(self.mocap_category)):
                c_idx = (latent_c_idx == i).cpu().numpy()
                traj_idxs[c_idx] = self.weighted_traj_idx_sample_batch(np.sum(c_idx), labeled=True, target_type=i)
        # traj_idxs = self.weighted_traj_idx_sample_batch(num_frames, labeled=True)
        times = self.traj_time_sample_batch(traj_idxs, labeled=True)
        return self.get_full_frame_at_time_batch(traj_idxs, times, labeled=True)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = MotionLoader.get_root_pos(frame0), MotionLoader.get_root_pos(frame1)
        root_rot0, root_rot1 = MotionLoader.get_root_rot(frame0), MotionLoader.get_root_rot(frame1)
        joints0, joints1 = MotionLoader.get_joint_pose(frame0), MotionLoader.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = MotionLoader.get_tar_toe_pos_local(frame0), MotionLoader.get_tar_toe_pos_local(
            frame1)
        linear_vel_0, linear_vel_1 = MotionLoader.get_linear_vel(frame0), MotionLoader.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = MotionLoader.get_angular_vel(frame0), MotionLoader.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = MotionLoader.get_joint_vel(frame0), MotionLoader.get_joint_vel(frame1)

        blend_root_pos = self.interp(root_pos0, root_pos1, blend)
        blend_root_rot = transformations.quaternion_slerp(
            root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot),
            dtype=torch.float32, device=self.device)
        blend_joints = self.interp(joints0, joints1, blend)
        blend_tar_toe_pos = self.interp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.interp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.interp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.interp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos, blend_root_rot, blend_joints, blend_tar_toe_pos,
            blend_linear_vel, blend_angular_vel, blend_joints_vel])

    def feed_forward_generator_lb(self, num_mini_batch, mini_batch_size):
        """Generate a batch of labeled demonstrations."""
        for _ in range(num_mini_batch):
            idx = np.random.choice(self.preloaded_s_lb.shape[0], size=mini_batch_size)
            s = self.preloaded_s_lb[idx, :]
            labels = self.preloaded_label[idx]
            yield s, labels

    def feed_forward_generator_ulb(self, num_mini_batch, mini_batch_size):
        """Generate a batch of labeled demonstrations."""
        for _ in range(num_mini_batch):
            idx = np.random.choice(self.preloaded_s_ulb.shape[0], size=mini_batch_size)
            s = self.preloaded_s_ulb[idx, :]
            yield s

    @staticmethod
    def get_root_pos(pose):
        return pose[MotionLoader.ROOT_POS_START_IDX:MotionLoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_pos_batch(poses):
        return poses[:, MotionLoader.ROOT_POS_START_IDX:MotionLoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_rot(pose):
        return pose[MotionLoader.ROOT_ROT_START_IDX:MotionLoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_root_rot_batch(poses):
        return poses[:, MotionLoader.ROOT_ROT_START_IDX:MotionLoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_joint_pose(pose):
        return pose[MotionLoader.JOINT_POSE_START_IDX:MotionLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_pose_batch(poses):
        return poses[:, MotionLoader.JOINT_POSE_START_IDX:MotionLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_tar_toe_pos_local(pose):
        return pose[MotionLoader.TAR_TOE_POS_LOCAL_START_IDX:MotionLoader.TAR_TOE_POS_LOCAL_END_IDX]

    @staticmethod
    def get_tar_toe_pos_local_batch(poses):
        return poses[:, MotionLoader.TAR_TOE_POS_LOCAL_START_IDX:MotionLoader.TAR_TOE_POS_LOCAL_END_IDX]

    @staticmethod
    def get_linear_vel(pose):
        return pose[MotionLoader.LINEAR_VEL_START_IDX:MotionLoader.LINEAR_VEL_END_IDX]

    @staticmethod
    def get_linear_vel_batch(poses):
        return poses[:, MotionLoader.LINEAR_VEL_START_IDX:MotionLoader.LINEAR_VEL_END_IDX]

    @staticmethod
    def get_angular_vel(pose):
        return pose[MotionLoader.ANGULAR_VEL_START_IDX:MotionLoader.ANGULAR_VEL_END_IDX]

    @staticmethod
    def get_angular_vel_batch(poses):
        return poses[:, MotionLoader.ANGULAR_VEL_START_IDX:MotionLoader.ANGULAR_VEL_END_IDX]

    @staticmethod
    def get_joint_vel(pose):
        return pose[MotionLoader.JOINT_VEL_START_IDX:MotionLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_joint_vel_batch(poses):
        return poses[:, MotionLoader.JOINT_VEL_START_IDX:MotionLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_tar_toe_vel_local(pose):
        return pose[MotionLoader.TAR_TOE_VEL_LOCAL_START_IDX:MotionLoader.TAR_TOE_VEL_LOCAL_END_IDX]

    @staticmethod
    def get_tar_toe_vel_local_batch(poses):
        return poses[:, MotionLoader.TAR_TOE_VEL_LOCAL_START_IDX:MotionLoader.TAR_TOE_VEL_LOCAL_END_IDX]
