from .base.legged_robot import LeggedRobot
from .go2.go2_agility_config import Go2AgilityCfg, Go2AgilityCfgPPO

from legged_gym.utils.task_registry import task_registry

task_registry.register("go2", LeggedRobot, Go2AgilityCfg(), Go2AgilityCfgPPO())
