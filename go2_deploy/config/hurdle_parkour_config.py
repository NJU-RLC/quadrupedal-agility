# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .highlevel_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2HurdleCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,  # [rad]
            'RL_hip_joint': 0.0,  # [rad]
            'FR_hip_joint': 0.0,  # [rad]
            'RR_hip_joint': 0.0,  # [rad]

            'FL_thigh_joint': 0.9,  # [rad]
            'RL_thigh_joint': 0.9,  # [rad]
            'FR_thigh_joint': 0.9,  # [rad]
            'RR_thigh_joint': 0.9,  # [rad]

            'FL_calf_joint': -1.8,  # [rad]
            'RL_calf_joint': -1.8,  # [rad]
            'FR_calf_joint': -1.8,  # [rad]
            'RR_calf_joint': -1.8,  # [rad]
        }
    class env(LeggedRobotCfg.env):
        num_envs = 14000
        n_scan = 132
        n_priv = 4
        n_delta_yaw = 2
        n_obst_type = 1
        n_auxiliary = n_delta_yaw + n_obst_type  # delta_yaws, one-hot vector for 6 obstacles
        n_priv_latent = 29
        n_proprio = 57 + n_auxiliary
        history_len = 10
        num_command = 5 + 1 + 2  # 5 commands, 1 eps, 3 cate
        num_obs_disc = 49
        disc_obs_len = 6  # discriminator observation horizon

        num_observations = n_proprio + n_scan + n_priv_latent + n_priv + history_len * (n_proprio - n_auxiliary)
        num_observations_ll = n_proprio - n_auxiliary + n_priv_latent + n_priv + num_command
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions_d = 2
        num_actions_c = 5 + 1  # 5 commands, 1 eps
        num_actions_ll = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        # mocap_category = ['walk', 'pace', 'trot', 'canter', 'jump']
        mocap_category = ['canter', 'jump']
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        obs_type = "og"
        history_encoding = True
        reorder_dofs = True
        # action_delay_range = [0, 5]
        # additional visual inputs
        # action_delay_range = [0, 5]
        # additional visual inputs
        include_foot_contacts = True

        randomize_start_pos = False
        randomize_start_vel = False
        randomize_start_yaw = False
        rand_yaw_range = 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = False
        rand_pitch_range = 1.6

        contact_buf_len = 100

        next_goal_threshold = 0.2
        # reach_goal_delay = 0.1
        reach_goal_delay = 0.02
        num_future_goal_obs = 2
        leave_goal_threshold = 4.0  # terminate when the distance to current goal is greater than 4m

        root_height_obs = True

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1}  # [N*m*s/rad]
        action_scale = 0.25
        action_bias_scale = 0.1
        hip_scale_reduction = 0.5
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        foot_name = "foot"
        penalize_contacts_on = [
            "base", "Head_upper", "Head_lower",
            "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
            "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]
        # penalize_contacts_on = []
        # terminate_after_contacts_on = []
        # terminate_after_contacts_on = [
        #     "base", "Head_upper", "Head_lower",
        #     "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
        #     "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]
        terminate_after_contacts_on = [
            "base", "Head_upper", "Head_lower", "hip", "thigh"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 5  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 0.02  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        lin_vel_clip = 0.2
        ang_vel_clip = 0.4



        class max_ranges:  # ['canter', 'jump']
            lin_vel_x = [[1.0, 3.5], [1.0, 3.5]]
            lin_vel_y = [[-0.5, 0.5], [-0.3, 0.3]]
            ang_vel_yaw = [[-0.5, 0.5], [-0.5, 0.5]]
            jump_height = [0.50, 0.55]
            locomotion_height = [0.25, 0.34]

        class crclm_incremnt:
            lin_vel_x = 0.1  # min max [m/s]
            lin_vel_y = 0.1  # min max [m/s]
            ang_vel_yaw = 0.1  # min max [rad/s]
            heading = 0.5

        waypoint_delta = 0.7


class Go2HurdleCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_a1'
        ll_policy_path = 'go2_deploy/models/ll_policy/hurdle/model.pt'
