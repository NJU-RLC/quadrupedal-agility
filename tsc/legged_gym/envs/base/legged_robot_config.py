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
        num_envs = 6144
        n_scan = 132
        n_priv = 4
        n_delta_yaw = 2
        n_obst_type = 6
        n_auxiliary = n_delta_yaw + n_obst_type  # delta_yaws, one-hot vector for 6 obstacles
        n_priv_latent = 29
        n_proprio = 57 + n_auxiliary
        history_len = 10
        mocap_category = ['trot', 'canter', 'jump']
        mocap_category_all = ['walk', 'pace', 'trot', 'canter', 'jump']
        num_actions_d = len(mocap_category)  # 3 selected behavior modes
        num_actions_c = 5 + 1  # 5 commands + 1 epsilon
        num_actions_bbc = 12
        num_command = num_actions_c + len(mocap_category)  # 5 commands + 1 epsilon + 3 selected behavior modes

        num_observations = n_proprio + n_scan + n_priv_latent + n_priv + history_len * (n_proprio - n_auxiliary)
        num_observations_bbc = (n_proprio - n_auxiliary + n_priv_latent + n_priv + num_actions_c +
                                len(mocap_category_all))
        num_privileged_obs = None
        num_obs_disc = 49
        disc_obs_len = 2  # discriminator observation horizon

        send_timeouts = True
        episode_length_s = 40  # episode length in seconds
        history_encoding = True
        include_foot_contacts = True
        env_spacing = 3.

        randomize_start_pos = False
        randomize_start_vel = True
        randomize_start_yaw = True
        rand_yaw_range = 0.2
        randomize_start_x = True
        rand_x_range = 0.2
        randomize_start_y = True
        rand_y_range = 0.1
        randomize_start_pitch = False
        rand_pitch_range = 1.6

        contact_buf_len = 100

        next_goal_threshold = 0.4
        reach_goal_delay = 0.02
        num_future_goal_obs = 2
        leave_goal_threshold = 4.0  # terminate when the distance to current goal is greater than 4m

        root_height_obs = True

    class depth:
        use_camera = False
        camera_num_envs = 256
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.305, 0.0175, 0.098]  # front camera go2
        angle = [-5, 5]  # positive pitch down

        update_interval = 1

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2

        near_clip = 0.3
        far_clip = 4
        depth_noise = 0.05

        scale = 1
        invert = True

    class normalization:
        class obs_scales:
            lin_vel = 0.5
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            key_pos = 0.0
            foot_contact = 0.0
            lin_vel_dist = 0.0
            ang_vel_dist = 0.0
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
        mesh_type = 'obstacle'
        hf2mesh_method = "grid"
        max_error = 0.1
        max_error_camera = 2

        y_range = [-0.4, 0.4]

        edge_width_thresh = 0.05
        horizontal_scale = 0.05
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005
        border_size = 5
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
        measure_horizontal_noise = 0.0
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 18.
        terrain_width = 4
        num_rows = 10
        num_cols = 40

        terrain_dict = {}
        terrain_proportions = list(terrain_dict.values())

        slope_treshold = 1.5
        origin_zero_z = True

    class obstacle:
        files = ['{LEGGED_GYM_ROOT_DIR}/resources/obstacles/bar_jump/bar_jump.urdf',
                 '{LEGGED_GYM_ROOT_DIR}/resources/obstacles/frame/frame.urdf',
                 '{LEGGED_GYM_ROOT_DIR}/resources/obstacles/poles/poles.urdf',
                 '{LEGGED_GYM_ROOT_DIR}/resources/obstacles/seesaw/seesaw.urdf',
                 '{LEGGED_GYM_ROOT_DIR}/resources/obstacles/tire_jump/tire_jump.urdf',
                 '{LEGGED_GYM_ROOT_DIR}/resources/obstacles/tunnel/tunnel.urdf']
        border_file = '{LEGGED_GYM_ROOT_DIR}/resources/obstacles/border/border.urdf'

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

        bar_jump_range = [0.05, 0.20]  # [0.05, 0.20], [0.05, 0.25], [0.05, 0.30]
        tire_jump_range = [0.40, 0.55]  # [0.40, 0.55], [0.40, 0.60], [0.05, 0.65]
        curriculum = False
        curr_step = 0.01
        curr_threshold = 0.8
        bar_jump_init_range = [0.05, 0.10]
        tire_jump_init_range = [0.40, 0.45]
        bar_jump_max_range = [0.05, 0.3]
        tire_jump_max_range = [0.40, 0.65]

        horizontal_scale = 0.05  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 5  # [m]
        border_height = 4.28  # 0.72m
        randomize_border = True
        border_height_range = [0.0, 4.5]
        env_length = 7
        env_width = 10
        env_boarder = 1.5
        robot_org = [4.5, 0.5]
        num_goals = 4
        last_goal_repeat = 2
        measure_heights = True
        measured_points_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        randomize_start = False

        num_obst_per_env = 6
        random_x = {"bar_jump": [-0.25, 0.25],
                    "frame": [-0.25, 0.25],
                    "poles": [-0.25, 0.25],
                    "seesaw": [-0.25, 0.25],
                    "tire_jump": [-0.25, 0.25],
                    "tunnel": [-0.25, 0.25]}
        random_y = [-0.15, 0.15]
        random_yaw = [-5, 5]

        frame_pos = [[[5.5, 1.0], [5.5, 5.0]], [[5.5, 5.0], [5.5, 9.0]], [[3.5, 9.0], [3.5, 5.0]],
                     [[3.5, 5.0], [3.5, 1.0]], [[1.5, 1.0], [1.5, 5.0]], [[1.5, 5.0], [1.5, 9.0]]]
        frame_ang = [90, 90, -90, -90, 90, 90]

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 5  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 0.02  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:  # ['walk', 'pace', 'trot', 'canter', 'jump']
            lin_vel_x = [[0.0, 0.6], [0.5, 1.5], [0.5, 1.5], [0.8, 2.5], [0.8, 2.0]]
            lin_vel_y = [[-0.15, 0.15], [-0.3, 0.3], [-0.3, 0.3], [-0.5, 0.5], [-0.3, 0.3]]
            ang_vel_yaw = [[-1.0, 1.0], [-1.57, 1.57], [-1.57, 1.57], [-0.5, 0.5], [-0.5, 0.5]]
            jump_height = [0.45, 0.58]
            locomotion_height = [0.25, 0.34]

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
        action_bbc_weight = 0.8
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

        randomize_base_mass = False
        added_mass_range = [0.0, 1.5]
        randomize_base_com = False
        added_com_range = [-0.1, 0.1]
        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.5

        randomize_action = True
        action_noise = [0.8, 1.2]
        randomize_motor = False
        motor_strength_range = [0.8, 1.2]

        action_delay = True
        action_delay_step = 1
        action_buf_len = 8

    class rewards:
        class scales:
            termination = -50.0
            reach_goal = 5.0
            every_step = 0.0
            # tracking rewards
            tracking_goal_vel = 0.4
            tracking_yaw = 2.0
            # regularization rewards
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            dof_acc = 0.0
            collision = -20.0
            action_rate = 0.0
            action_hl_rate = -0.2
            latent_c_rate = -1.0
            delta_torques = 0.0
            torques = 0.0
            hip_pos = 0.0
            dof_error = 0.0
            feet_stumble = 0.0
            feet_edge = -1.0
            torque_limits = 0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1
        soft_torque_limit = 0.4
        base_height_target = 1.
        max_contact_force = 40.
        target_lin_vel = 0.4

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2


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

        tanh_encoder_output = False

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 5.e-4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 500, 1000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]

    class depth_encoder:
        if_depth = LeggedRobotCfg.depth.use_camera
        depth_shape = LeggedRobotCfg.depth.resized
        buffer_len = LeggedRobotCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        learning_rate_byol = 3.e-4
        learning_rate_min = 1.e-5
        num_steps_per_env = LeggedRobotCfg.depth.update_interval * 24

    class estimator:
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = LeggedRobotCfg.env.n_priv
        num_prop = LeggedRobotCfg.env.n_proprio - LeggedRobotCfg.env.n_auxiliary
        num_auxiliary = LeggedRobotCfg.env.n_auxiliary
        num_scan = LeggedRobotCfg.env.n_scan
        load_estimator_bbc = True

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 50000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'agility'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        bbc_path = ''

        # Discriminator
        disc_loss_function = 'MSELoss'  # BCEWithLogitsLoss, MSELoss, WassersteinLoss
        reward_i_coef = 0.05
        reward_us_coef = 0.0
        reward_ss_coef = 0.0
        reward_t_coef = 2.0
        disc_hidden_units = [512, 256]
