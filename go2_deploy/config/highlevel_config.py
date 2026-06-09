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

from posixpath import relpath
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from .base_config import BaseConfig
import torch.nn as nn


class LeggedRobotCfg(BaseConfig):
    class play:
        load_student_config = False
        mask_priv_obs = False

    class env:
        num_envs = 14000
        n_scan = 132
        n_priv = 4
        n_delta_yaw = 2
        n_obst_type = 6
        n_auxiliary = n_delta_yaw + n_obst_type  # delta_yaws, one-hot vector for 6 obstacles
        n_priv_latent = 29
        n_proprio = 57 + n_auxiliary
        history_len = 10
        num_command = 5 + 1 + 3  # 5 commands, 1 eps, 3 cate
        num_obs_disc = 45
        disc_obs_len = 6  # discriminator observation horizon

        num_observations = n_proprio + n_scan + n_priv_latent + n_priv + history_len * (n_proprio - n_auxiliary)
        num_observations_ll = n_proprio - n_auxiliary + n_priv_latent + n_priv + num_command
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions_d = 3
        num_actions_c = 5 + 1  # 5 commands, 1 eps
        num_actions_ll = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        # mocap_category = ['walk', 'pace', 'trot', 'canter', 'jump']
        mocap_category = ['trot', 'canter', 'jump']
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

    class depth:
        use_camera = False#True
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        # position = [0.27, 0, 0.03]  # front camera a1
        position = [0.33, 0, 0.043]  # front camera go2
        angle = [-5, 5]  # positive pitch down

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2

        near_clip = 0
        far_clip = 3  # far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

    class normalization:
        class obs_scales:
            lin_vel = 0.5
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values
        quantize_height = True

        class noise_scales:
            rotation = 0.0
            dof_pos = 0.01
            dof_vel = 0.05
            lin_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.02

    class terrain:
        mesh_type = 'obstacle'  # "heightfield" # none, plane, obstacle, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1  # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]

        edge_width_thresh = 0.05
        horizontal_scale = 0.05  # [m] influence computation time by a lot
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005  # [m]
        border_size = 5  # [m]
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = True

        all_vertical = False
        no_flat = True

        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05,
                             1.2]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.0

        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 18.
        terrain_width = 4
        num_rows = 10  # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 40  # number of terrain cols (types)

        terrain_dict = {"smooth slope": 0.,
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0.,
                        "rough stairs down": 0.,
                        "discrete": 0.,
                        "stepping stones": 0.0,
                        "gaps": 0.,
                        "smooth flat": 0,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.2,
                        "parkour_hurdle": 0.2,
                        "parkour_flat": 0.2,
                        "parkour_step": 0.2,
                        "parkour_gap": 0.2,
                        "demo": 0.0, }
        terrain_proportions = list(terrain_dict.values())

        # trimesh only:
        slope_treshold = 1.5  # slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True

        num_goals = 8

    class obstacle:
        files = ['{LEGGED_GYM_ROOT_DIR}/resources/obstacles/bar_jump/bar_jump.urdf',
                 '{LEGGED_GYM_ROOT_DIR}/resources/obstacles/frame/frame.urdf',
                 '{LEGGED_GYM_ROOT_DIR}/resources/obstacles/poles/poles.urdf',
                 '{LEGGED_GYM_ROOT_DIR}/resources/obstacles/seesaw/seesaw.urdf',
                 '{LEGGED_GYM_ROOT_DIR}/resources/obstacles/tire_jump/tire_jump.urdf',
                 '{LEGGED_GYM_ROOT_DIR}/resources/obstacles/tunnel/tunnel.urdf']

        obstacle_dict = {"bar_jump": 0.2,
                         "frame": 0.15,
                         "poles": 0.2,
                         "seesaw": 0.15,
                         "tire_jump": 0.2,
                         "tunnel": 0.1}
        obstacle_proportions = list(obstacle_dict.values())

        num_links = {"bar_jump": 2,
                     "frame": 1,
                     "poles": 1,
                     "seesaw": 2,
                     "tire_jump": 2,
                     "tunnel": 1}
        num_obstacle_links = list(num_links.values())

        num_joints = {"bar_jump": 1,
                      "frame": 0,
                      "poles": 0,
                      "seesaw": 1,
                      "tire_jump": 1,
                      "tunnel": 0}
        num_obstacle_joints = list(num_joints.values())

        bar_jump_range = [0.1, 0.2]
        tire_jump_range = [0.45, 0.55]

        horizontal_scale = 0.05  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 5  # [m]
        curriculum = True
        env_length = 10
        env_width = 8
        num_goals = 8
        last_goal_repeat = 2
        measure_heights = True
        measured_points_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]  # 1m x 1.1m
        # measured_points_x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        random_start_x = [1.5, 3.0]  # TODO: far_clip = 2

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 5  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 0.02  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        lin_vel_clip = 0.2
        ang_vel_clip = 0.4

        # class ranges:  # ['walk', 'pace', 'trot', 'canter', 'jump']
        #     lin_vel_x = [[0.0, 0.5], [0.5, 1.0], [0.5, 1.0], [0.8, 1.2], [0.8, 1.2]]
        #     lin_vel_y = [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]
        #     ang_vel_yaw = [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]
        #     jump_height = [0.50, 0.50]
        #     locomotion_height = [0.30, 0.32]
        #
        # class max_ranges:  # ['walk', 'pace', 'trot', 'canter', 'jump']
        #     lin_vel_x = [[0.0, 0.5], [0.5, 1.5], [0.5, 1.5], [0.8, 2.0], [0.8, 1.6]]
        #     lin_vel_y = [[-0.15, 0.15], [-0.3, 0.3], [-0.3, 0.3], [-0.5, 0.5], [-0.3, 0.3]]
        #     ang_vel_yaw = [[-1.0, 1.0], [-1.57, 1.57], [-1.57, 1.57], [-0.5, 0.5], [-0.5, 0.5]]
        #     jump_height = [0.50, 0.75]
        #     locomotion_height = [0.28, 0.38]

        class ranges:  # ['trot', 'canter', 'jump']
            lin_vel_x = [[0.5, 1.0], [0.8, 1.2], [0.8, 1.5]]
            lin_vel_y = [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]
            ang_vel_yaw = [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]
            jump_height = [0.45, 0.55]
            locomotion_height = [0.25, 0.34]

        class max_ranges:  # ['trot', 'canter', 'jump']
            lin_vel_x = [[0.5, 1.5], [1.2, 2.5], [1.6, 2.0]]
            lin_vel_y = [[-0.3, 0.3], [-0.5, 0.5], [-0.3, 0.3]]
            ang_vel_yaw = [[-1.57, 1.57], [-0.5, 0.5], [-0.5, 0.5]]
            jump_height = [0.50, 0.58]
            locomotion_height = [0.25, 0.34]

        # class max_ranges:  # ['walk', 'pace', 'trot', 'canter', 'jump']
        #     lin_vel_x = [[0.5, 1.8], [0.8, 2.5], [0.8, 2.0]]
        #     lin_vel_y = [[-0.5, 0.5], [-0.5, 0.5], [-0.3, 0.3]]
        #     ang_vel_yaw = [[-1.57, 1.57], [-0.5, 0.5], [-0.5, 0.5]]
        #     jump_height = [0.45, 0.58]
        #     locomotion_height = [0.25, 0.34]

        class crclm_incremnt:
            lin_vel_x = 0.1  # min max [m/s]
            lin_vel_y = 0.1  # min max [m/s]
            ang_vel_yaw = 0.1  # min max [rad/s]
            heading = 0.5

        waypoint_delta = 0.7

    class init_state:
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "joint_a": 0.,
            "joint_b": 0.}

    class control:
        control_type = 'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        action_ll_weight = 0.8
        hip_scale_reduction = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.6, 2.0]
        randomize_base_mass = False  # TODO
        added_mass_range = [0.0, 3.0]
        randomize_base_com = False  # TODO
        added_com_range = [-0.2, 0.2]
        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.5

        randomize_motor = False
        motor_strength_range = [0.8, 1.2]

        delay_update_global_steps = 24 * 8000
        action_delay = True
        action_curr_step = [1, 1]
        action_curr_step_scratch = [1, 1]
        action_delay_view = 1
        action_buf_len = 8

    class rewards:
        class scales:
            termination = -20.0
            reach_goal = 10.0
            # tracking rewards
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            # regularization rewards
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            dof_acc = -2.5e-7
            collision = -10.
            action_rate = -0.1
            delta_torques = -1.0e-7
            torques = -0.00001
            hip_pos = -0.5
            dof_error = -0.04
            feet_stumble = -1
            feet_edge = -1
            torque_limits = 0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1
        soft_torque_limit = 0.4
        base_height_target = 1.
        max_contact_force = 40.  # forces above this value are penalized

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [64]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        init_with_orthogonal =  False

        tanh_encoder_output = False

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]

    class depth_encoder:
        if_depth = LeggedRobotCfg.depth.use_camera
        depth_shape = LeggedRobotCfg.depth.resized
        buffer_len = LeggedRobotCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = LeggedRobotCfg.depth.update_interval * 24

    class estimator:
        init_with_orthogonal = False

        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = LeggedRobotCfg.env.n_priv
        num_prop = LeggedRobotCfg.env.n_proprio - LeggedRobotCfg.env.n_auxiliary
        num_auxiliary = LeggedRobotCfg.env.n_auxiliary
        num_scan = LeggedRobotCfg.env.n_scan

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 50000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'rough_a1'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        ll_policy_path = ''

        # Discriminator
        disc_loss_function = 'MSELoss'  # [BCEWithLogitsLoss, MSELoss, WassersteinLoss]
        reward_i_coef = 0.0
        reward_us_coef = 0.0  # 0.05-0.5之间
        reward_ss_coef = 0.0  # 0.05, 0.1, 0.2
        reward_t_coef = 1.0  # 0.2
        disc_hidden_units = [512, 256]
