from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2AgilityCfg(LeggedRobotCfg):
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
        terminate_after_contacts_on = [
            "base", "Head_upper", "Head_lower", "hip", "thigh"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25


class Go2AgilityCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'agility'
        bbc_path = 'weights/bbc/model.pt'
