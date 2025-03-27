import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ROOT_DIR)
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import numpy as np


def train(args):
    if args.debug:
        args.num_envs = 64
        args.headless = False
    else:
        args.headless = True

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    train(args)
