import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ROOT_DIR)
import isaacgym
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import torch.nn.functional as F
import statistics
import numpy as np
from collections import deque


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1024
    env_cfg.env.episode_length_s = 40

    env_cfg.depth.angle = [0, 1]
    env_cfg.depth.depth_noise = 0.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_action = False
    env_cfg.obstacle.randomize_start = True
    env_cfg.obstacle.curriculum = False
    env_cfg.obstacle.randomize_border = False
    env_cfg.env.next_goal_threshold = 0.45

    reach_goal_buffer = deque(maxlen=1000)

    if args.use_camera:
        env_cfg.env.num_envs = 256
        env_cfg.env.draw_height_maps = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    obs_bbc = env.get_observations_bbc()

    # load policy
    train_cfg.runner.resume = True
    train_cfg.estimator.load_estimator_bbc = False
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                                                   train_cfg=train_cfg, return_log_dir=True)

    policy = ppo_runner.get_inference_policy(device=env.device)
    policy_bbc = ppo_runner.get_inference_policy_bbc(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        ppo_runner.depth_backbone.augment = None
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None
    infos["delta_yaw_ok"] = torch.ones(env.num_envs, dtype=torch.bool, device=ppo_runner.device)

    n_proprio = env.cfg.env.n_proprio
    n_delta_yaw = env.cfg.env.n_delta_yaw
    n_obst_type = env.cfg.env.n_obst_type
    n_auxiliary = env.cfg.env.n_auxiliary
    n_depth_latent = train_cfg.policy.scan_encoder_dims[-1]

    estimator_cfg = train_cfg.estimator
    n_priv_est = estimator_cfg.priv_states_dim
    n_prop_est = estimator_cfg.num_prop
    n_scan_est = estimator_cfg.num_scan
    delta_yaw_set = []
    delta_yaw_pred_set = []
    abs_vel_set = []
    for i in range(1000 * int(env.max_episode_length)):
        with torch.no_grad():
            priv_states_estimated = estimator(obs[:, :n_prop_est])
            obs[:, n_prop_est + n_auxiliary + n_scan_est:n_prop_est + n_auxiliary + n_scan_est + n_priv_est] = priv_states_estimated
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, :n_proprio].clone()
                    obs_student[:, n_proprio-n_auxiliary:n_proprio] = 0
                    depth_latent_yaw_type = depth_encoder(infos["depth"], obs_student)
                    depth_latent = depth_latent_yaw_type[:, :n_depth_latent]
                    delta_yaw = depth_latent_yaw_type[:, n_depth_latent:n_depth_latent + n_delta_yaw]
                    obst_type = depth_latent_yaw_type[:, n_depth_latent + n_delta_yaw:]

                obs[:, n_proprio - n_auxiliary:n_proprio - n_auxiliary + n_delta_yaw] = 1.5 * delta_yaw
                obs[:, n_proprio - n_auxiliary + n_delta_yaw:n_proprio] = \
                    F.one_hot(torch.argmax(obst_type.detach(), dim=-1), num_classes=obst_type.shape[-1])
                pred_obst_type = torch.argmax(obst_type.detach(), dim=-1)[env.lookat_id].item()
                pred_obst_type = env.obstacle.obst_types[int(pred_obst_type)]
                print("pred_obst_type: {}".format(pred_obst_type))

                delta_yaw_set.append(env.delta_yaw[0].cpu().numpy())
                delta_yaw_pred_set.append(delta_yaw[0].cpu().numpy())
                abs_vel_set.append((env.base_lin_vel[0][0].item()**2 + env.base_lin_vel[0][1].item()**2)**0.5)
            else:
                depth_latent = None

            if hasattr(ppo_runner.alg, "depth_actor"):
                embedding = ppo_runner.alg.depth_actor(obs, hist_encoding=True, scandots_latent=depth_latent)
                prob = ppo_runner.alg.depth_actor.actor_d(embedding)
                mean = ppo_runner.alg.depth_actor.actor_c(embedding)
                actions = torch.cat([torch.argmax(prob, dim=-1).unsqueeze(-1), mean], dim=-1)

            else:
                actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)

            next_commands = env.set_commands(actions)
            obs_bbc[:, -next_commands.shape[1]:] = next_commands
            actions_bbc = policy_bbc(obs_bbc, hist_encoding=True)

            obs, _, rews, dones, infos, _, _ = env.step(actions_bbc)
            obs_bbc = env.get_observations_bbc()

            cur_obstacle_type = env.cur_obstacle_types[env.lookat_id].item()
            cur_obstacle_type = env.obstacle.obst_types[int(cur_obstacle_type)]
            new_ids = (dones > 0).nonzero(as_tuple=False)
            reach_goal_buffer.extend(infos['reach_goal'][new_ids][:, 0].cpu().numpy().tolist())
            print("time: {} ".format(int(env.episode_length_buf[env.lookat_id].item() / 50)),
                  "cmd_vx: {} ".format(np.round(env.commands[env.lookat_id, 0].item(), 2)),
                  "actual_vx: {} ".format(np.round(env.base_lin_vel[env.lookat_id, 0].item(), 2)),
                  "cur_obst_type: {} ".format(cur_obstacle_type),
                  "latent_eps: {} ".format(np.round(env.latent_eps[env.lookat_id].item(), 2)),
                  "latent_c: {} ".format(torch.argmax(env.latent_c[env.lookat_id]).item()))
            if len(reach_goal_buffer) > 0:
                print("success_rate: {} ".format(np.round(statistics.mean(reach_goal_buffer), 4)))


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
