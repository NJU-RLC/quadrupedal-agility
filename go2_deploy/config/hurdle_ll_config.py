import glob
from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgAlgo

MOTION_FILES_LB = glob.glob('../../mocap_data/mocap_canter/*')
MOTION_FILES_ULB = glob.glob('../../mocap_data/mocap_canter/*')


class Go2HurdleLLCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None
        num_prop = 57
        num_explicit = 4
        num_latent = 29
        num_command = 9
        num_obs = num_prop + num_explicit + num_latent + num_command
        num_privileged_obs = num_obs
        num_obs_disc = 46
        mocap_state_init = False
        recovery_init_prob = 0.0
        motion_files_lb = MOTION_FILES_LB
        motion_files_ulb = MOTION_FILES_ULB
        mocap_category = ['canter']
        episode_length_s = 20.
        root_height_obs = True
        history_len = 10
        contact_buf_len = 100

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
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

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # 40
        damping = {'joint': 1.}  # 0.5
        action_scale = 0.25  # 0.75
        hip_scale_reduction = 0.3#0.36 #0.5
        decimation = 4  # Number of control action updates @ sim DT per policy DT

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class asset(LeggedRobotCfg.asset):
        name = "legged_robot"
        foot_name = "foot"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        # penalize_contacts_on = []
        penalize_contacts_on = ["thigh", "calf", "base"]
        # terminate_after_contacts_on = [
        #     "base", "Head_upper", "Head_lower",
        #     "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
        #     "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable, bitwise filter

    class domain_rand:
        randomize_friction = True
        friction_range = [0.6, 2.0]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        randomize_base_com = True
        added_com_range = [-0.2, 0.2]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5
        # randomize_gains = True
        # stiffness_multiplier_range = [0.9, 1.1]
        # damping_multiplier_range = [0.9, 1.1]

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]
        action_delay = True
        action_buf_len = 8
        delay_update_global_steps = 24 * 2000
        action_curr_step = [0, 1]

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            roll_pitch = 0.01
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class normalization:
        class obs_scales:
            lin_vel = 0.5
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100 #1.2

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        jump_goal = 100.

        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0.0
            tracking_lin_vel = 1.0  # 1.0
            tracking_ang_vel = 0.5  # 0.2
            jump_up_height = 0.0  # 2.0
            locomotion_height = 0.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            torques = -0.00001
            delta_torques = -1.0e-7
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = 0.0
            feet_air_time = 0.0
            collision = -10.
            feet_stumble = -1.
            action_rate = -0.1
            stand_still = 0.0
            dof_pos_limits = 0.0
            hip_pos = -0.5
            dof_error = -0.04

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 5
        resampling_time = 6.  # time before command are changed [s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:  # ['walk', 'pace', 'trot', 'canter', 'jump']
            lin_vel_x = [[0.8, 1.5]]
            lin_vel_y = [[-0.3, 0.3]]
            ang_vel_yaw = [[-0.5, 0.5]]
            jump_height = [0.0, 0.0]
            locomotion_height = [0.0, 0.0]

        lin_vel_x_clip = 0.2
        lin_vel_y_clip = 0.05
        ang_vel_yaw_clip = 0.05


class Go2CanterCfgAlgo(LeggedRobotCfgAlgo):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [64]
        activation = 'elu'
        train_with_estimated_latent = True

    class algorithm(LeggedRobotCfgAlgo.algorithm):
        lr_ac = 1e-3
        lr_disc = 5e-4
        lr_q = 1e-3
        surrogate_loss_coef = 2.
        value_loss_coef = 5.
        entropy_coef = 0.01
        bounds_loss_coef = 0.0005  # <0.001
        disc_coef = 1.
        disc_logit_reg = 0.05
        disc_grad_penalty = 0.1  # TODO: 高动态的mocap需要较小的penalty 0.05, 0.1, 0.3
        disc_weight_decay = 0.0001
        disc_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

        us_coef = 1.0
        ss_coef = 1.0
        prior_soft_coef = 1e-3
        info_max_coef = 1.0
        begin_rim = 200
        disc_loss_function = 'MSELoss'  # [BCEWithLogitsLoss, MSELoss, WassersteinLoss]

        priv_reg_coef_schedual = [0, 0.1, 1000, 2000]  # [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]  # [0, 0.1, 0, 1]

    class estimator:
        train_with_estimated_explicit = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]

    class runner(LeggedRobotCfgAlgo.runner):
        experiment_idx = 0
        experiment_name = 'go2_canter'
        algorithm_class_name = 'GAIL'
        policy_class_name = 'ActorCritic'
        max_iterations = 500000  # 500000 # number of policy updates
        grad_penalty_update_interval = 500
        dagger_update_freq = 20

        motion_files_lb = MOTION_FILES_LB
        motion_files_ulb = MOTION_FILES_ULB
        num_preload_transitions = 200000
        reward_i_coef = 1.0
        reward_us_coef = 0.0  # 0.05-0.5之间
        reward_ss_coef = 0.0  # 0.05, 0.1, 0.2
        reward_t_coef = 0.3  # 0.2
        disc_hidden_units = [512, 256]

        min_normalized_std = [0.05, 0.02, 0.05] * 4