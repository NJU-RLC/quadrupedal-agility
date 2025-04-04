import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Random seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_sim_params(args, cfg):
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.action_scale is not None:
            env_cfg.control.action_scale = args.action_scale
        if args.tracking_lin_vel is not None:
            env_cfg.rewards.scales.tracking_lin_vel = args.tracking_lin_vel
        if args.tracking_ang_vel is not None:
            env_cfg.rewards.scales.tracking_ang_vel = args.tracking_ang_vel
        if args.jump_up_height is not None:
            env_cfg.rewards.scales.jump_up_height = args.jump_up_height
        if args.locomotion_height is not None:
            env_cfg.rewards.scales.locomotion_height = args.locomotion_height
        if args.disc_history_len is not None:
            env_cfg.env.disc_history_len = int(args.disc_history_len)
        if args.disc_obs_len is not None:
            env_cfg.env.disc_obs_len = int(args.disc_obs_len)
        if args.obs_disc_weight_step is not None:
            env_cfg.env.obs_disc_weight_step = args.obs_disc_weight_step
        if args.task_obs_weight_decay_steps is not None:
            env_cfg.normalization.task_obs_weight_decay_steps = args.task_obs_weight_decay_steps
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
            cfg_train.algorithm.priv_reg_coef_schedual = cfg_train.algorithm.priv_reg_coef_schedual_resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
        if args.reward_i_coef is not None:
            cfg_train.runner.reward_i_coef = args.reward_i_coef
        if args.reward_us_coef is not None:
            cfg_train.runner.reward_us_coef = args.reward_us_coef
        if args.reward_ss_coef is not None:
            cfg_train.runner.reward_ss_coef = args.reward_ss_coef
        if args.reward_t_coef is not None:
            cfg_train.runner.reward_t_coef = args.reward_t_coef
        if args.experiment_idx is not None:
            cfg_train.runner.experiment_idx = args.experiment_idx
        if args.us_coef is not None:
            cfg_train.algorithm.us_coef = args.us_coef
        if args.ss_coef is not None:
            cfg_train.algorithm.ss_coef = args.ss_coef
        if args.lr_ac is not None:
            cfg_train.algorithm.lr_ac = args.lr_ac
        if args.lr_q is not None:
            cfg_train.algorithm.lr_q = args.lr_q
        if args.disc_grad_penalty is not None:
            cfg_train.algorithm.disc_grad_penalty = args.disc_grad_penalty
        if args.lr_disc is not None:
            cfg_train.algorithm.lr_disc = args.lr_disc
        if args.disc_loss_function is not None:
            cfg_train.algorithm.disc_loss_function = args.disc_loss_function
        if args.bounds_loss_coef is not None:
            cfg_train.algorithm.bounds_loss_coef = args.bounds_loss_coef

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2_locomotion", "help": "Name of the task"},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,
         "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--load_run", "type": str, "default": "-1",
         "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,
         "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},

        {"name": "--headless", "type": bool, "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--device", "type": str, "default": 'gpu', "help": "cpu or gpu"},
        {"name": "--device_id", "type": int, "default": 0, "help": "GPU device id"},
        {"name": "--num_envs", "type": int,
         "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int,
         "help": "Maximum number of training iterations. Overrides config file if provided."},

        {"name": "--reward_i_coef", "type": float},
        {"name": "--reward_us_coef", "type": float},
        {"name": "--reward_ss_coef", "type": float},
        {"name": "--reward_t_coef", "type": float},
        {"name": "--us_coef", "type": float},
        {"name": "--ss_coef", "type": float},
        {"name": "--lr_ac", "type": float},
        {"name": "--lr_disc", "type": float},
        {"name": "--lr_q", "type": float},
        {"name": "--disc_grad_penalty", "type": float},
        {"name": "--action_scale", "type": float},
        {"name": "--disc_loss_function", "type": str},
        {"name": "--tracking_lin_vel", "type": float},
        {"name": "--tracking_ang_vel", "type": float},
        {"name": "--jump_up_height", "type": float},
        {"name": "--locomotion_height", "type": float},
        {"name": "--bounds_loss_coef", "type": float},
        {"name": "--disc_history_len", "type": float},
        {"name": "--disc_obs_len", "type": float},
        {"name": "--obs_disc_weight_step", "type": float},
        {"name": "--task_obs_weight_decay_steps", "type": int},
        {"name": "--experiment_idx", "type": int, "default": -1}
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="GAIL",
        custom_parameters=custom_parameters)

    if args.device == 'gpu':
        args.rl_device = "cuda:{}".format(args.device_id)
        args.sim_device = "cuda:{}".format(args.device_id)
        args.compute_device_id = args.device_id
        args.sim_device_id = args.device_id
        args.use_gpu_pipeline = True
    else:
        args.rl_device = args.device
        args.sim_device = args.device
        args.use_gpu_pipeline = False
    return args


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
