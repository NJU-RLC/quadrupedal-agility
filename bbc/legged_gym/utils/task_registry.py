import os
import torch
import numpy as np
from datetime import datetime
from typing import Tuple

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgAlgo


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgAlgo):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgAlgo]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered name or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym command line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name, self.task_classes: init in envs.__init__.py
        if name in self.task_classes:
            task_class = self.get_task_class(name)  # legged_robot.LeggedRobot()
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files  return env_config and train_config
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)  # return env_config and train_config
        set_seed(env_cfg.seed)  # seed = 1
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)  # 传递sim_params到issacgym
        env = task_class(cfg=env_cfg,  # legged_robot.py
                         sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device,
                         headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[
        OnPolicyRunner, LeggedRobotCfgAlgo]:
        """ Creates the training algorithm either from a registered name or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root == "default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            # log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
            log_dir = os.path.join(log_root, "{}".format(int(train_cfg.runner.experiment_idx)))
        elif log_root is None:
            log_dir = None
        else:
            # log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
            log_dir = os.path.join(log_root, "{}".format(int(train_cfg.runner.experiment_idx)))

        # print(train_cfg.runner_class_name)
        runner_class = eval(train_cfg.runner_class_name)
        train_cfg_dict = class_to_dict(train_cfg)
        runner = runner_class(env, train_cfg_dict, log_dir, device=args.rl_device)
        # save resume path before creating a new log_dir
        resume = train_cfg.runner.resume  # False
        if resume:
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run,
                                        checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        if hasattr(train_cfg.runner, 'pretrained_ll'):
            if train_cfg.runner.pretrained_ll:
                resume_path = get_load_path(log_root, load_run=train_cfg.runner.pretrained_ll,
                                            checkpoint=train_cfg.runner.checkpoint)
                print(f"Loading low level model from: {resume_path}")
                runner.load_ll(resume_path)
        path_set = []
        if len(train_cfg.runner.pre_trained_actor_path):
            for path in train_cfg.runner.pre_trained_actor_path:
                path_set.append(os.path.join(LEGGED_GYM_ROOT_DIR, path))
            print(f"Loading pre trained actor from: {path_set}")
            runner.load_pre_trained_actor(path_set)
        return runner, train_cfg


# make global task registry
task_registry = TaskRegistry()
