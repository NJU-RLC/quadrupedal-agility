import os
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry
from .base.legged_robot import LeggedRobot
from .go2.go2_locomotion_config import Go2LocomotionCfg, Go2LocomotionCfgAlgo

task_registry.register("go2_locomotion", LeggedRobot, Go2LocomotionCfg(), Go2LocomotionCfgAlgo())
