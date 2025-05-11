import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ROOT_DIR)
import time
import isaacgym
import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.torch_jit_utils import *


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_motor = True
    env_cfg.domain_rand.action_delay = True
    env_cfg.domain_rand.action_curr_step = [1]
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 1e10
    env_cfg.env.episode_length_s = 500.
    env_cfg.env.mocap_state_init = False
    env_cfg.env.recovery_init_prob = 0.
    env_cfg.env.root_height_obs = True

    train_cfg.runner.num_preload_transitions = 1
    train_cfg.policy.train_with_estimated_latent = True
    train_cfg.estimator.train_with_estimated_explicit = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    env_cfg.env.play_mode = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    stop_state_log = -1  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_pos_bias = np.array([-2.8, -1.8, 1.8])
    img_idx = 0

    # set initial commands
    env.latent_c[:, :] = 0
    env.commands[:, :] = 0
    env.latent_c[:, 2] = 1
    env.commands[:, 0] = 2

    for i in range(1000 * int(env.max_episode_length)):
        time0 = time.time()
        if not env_cfg.env.root_height_obs:
            obs[:, env_cfg.env.num_prop] = 0.0
        actions = policy(obs.detach())
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                        'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            robot_pos = env.root_states[0, :3].cuda().cpu().numpy()
            camera_position = (trans_matrix_ba(robot_pos, [0, 0, 0]) @ np.append(camera_pos_bias, 1))[:-1]
            env.set_camera(camera_position, robot_pos)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': env.default_dof_pos[0, :].cpu().numpy() + env.cfg.control.action_scale *
                                      env.actions[robot_index, :].cpu().numpy(),
                    'dof_pos': env.dof_pos[robot_index, :].cpu().numpy(),
                    'dof_vel': env.dof_vel[robot_index, :].cpu().numpy(),
                    'dof_torque': env.torques[robot_index, :].cpu().numpy(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
            logger.plot_dof_pos()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)

        time1 = time.time()
        dt_target = 4 * 1 / 200
        dt_it = time1 - time0
        if dt_it < dt_target:
            time.sleep(dt_target - dt_it)


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
