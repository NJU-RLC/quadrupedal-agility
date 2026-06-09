import argparse
from pathlib import Path


DEFAULT_LCM_URL = "udpm://239.255.76.67:7667?ttl=255"
DEFAULT_LOG_ROOT = Path("go2_deploy/logs")
DEFAULT_RECOVER_POLICY = Path("go2_deploy/models/recover_policy/1_model/model.pt")


def add_runtime_args(parser, default_experiment_name):
    parser.add_argument("max_steps", nargs="?", type=int, default=10000000)
    parser.add_argument("--experiment-name", default=default_experiment_name)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lcm-url", default=DEFAULT_LCM_URL)
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--max-yaw-vel", type=float, default=1.0)
    parser.add_argument("--float32", action="store_true", help="Disable half precision. For inspection only.")
    return parser


def add_low_level_args(parser, default_ll_policy):
    parser.add_argument("--ll-policy", type=Path, default=default_ll_policy)
    parser.add_argument("--recover-policy", type=Path, default=DEFAULT_RECOVER_POLICY)
    return parser


def add_depth_args(parser, default_ll_policy, default_high_policy):
    parser.add_argument("--ll-policy", type=Path, default=default_ll_policy)
    parser.add_argument("--high-policy", type=Path, default=default_high_policy)
    parser.add_argument("--recover-policy", type=Path, default=DEFAULT_RECOVER_POLICY)
    parser.set_defaults(fix_high_estimator=True)
    parser.add_argument(
        "--fix-high-estimator",
        dest="fix_high_estimator",
        action="store_true",
        help="Use low-level estimator output for high-level explicit state. Enabled by default.",
    )
    parser.add_argument(
        "--no-fix-high-estimator",
        dest="fix_high_estimator",
        action="store_false",
        help="Use the high-level estimator output instead. This differs from the old FIX_MODE=True scripts.",
    )
    return parser


def ensure_device(device):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu --float32 only for inspection.")


def require_policy_file(policy_path, label):
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def build_low_level_model(mode_count, device, use_half):
    from go2_deploy.module.estimator import Estimator
    from go2_deploy.module.net_ll import ActorCritic

    num_command = 5 + 1 + mode_count
    model_ll = ActorCritic(
        num_actor_obs=57 + 4 + 29 + num_command,
        num_critic_obs=570 + 57 + 4 + 29 + num_command,
        num_actions=12,
        num_prop=57,
        num_hist=10,
        num_explicit=4,
        num_latent=29,
        num_command=num_command,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        priv_encoder_dims=[64],
        train_with_estimated_latent=True,
    ).to(device)
    estimator = Estimator(input_dim=57, output_dim=4, hidden_dims=[128, 64], device=device).to(device)
    if use_half:
        model_ll = model_ll.half()
        estimator = estimator.half()
    return model_ll, estimator


def validate_low_level_checkpoint(checkpoint, mode_count, check_classifier=True):
    actor_state = checkpoint.get("actor_critic", {})
    estimator_state = checkpoint.get("estimator", {})
    actor_weight = actor_state.get("actor_trunk.0.weight")
    classifier_weight = checkpoint.get("disc", {}).get("classifier.weight")
    estimator_weight = estimator_state.get("estimator.0.weight")

    expected_actor_in = 57 + 4 + 29 + (5 + 1 + mode_count)
    if actor_weight is not None and actor_weight.shape[1] != expected_actor_in:
        inferred_mode_count = actor_weight.shape[1] - 57 - 4 - 29 - 5 - 1
        raise ValueError(
            "Low-level checkpoint does not match this entry script. "
            f"actor_trunk.0.weight expects {actor_weight.shape[1]} inputs, "
            f"which implies mode_count={inferred_mode_count}; this script uses mode_count={mode_count}."
        )
    if check_classifier and classifier_weight is not None and classifier_weight.shape[0] != mode_count:
        raise ValueError(
            "Low-level checkpoint discriminator mode count does not match this entry script. "
            f"classifier.weight has {classifier_weight.shape[0]} rows; this script uses mode_count={mode_count}."
        )
    if estimator_weight is not None and estimator_weight.shape != (128, 57):
        raise ValueError(f"Unexpected estimator.0.weight shape: {tuple(estimator_weight.shape)}")


def load_recovery_policy(policy_path, device, use_half):
    import torch

    if not policy_path:
        return None, None
    policy_path = Path(policy_path)
    if not policy_path.exists():
        return None, None

    model, estimator = build_low_level_model(1, device, use_half)
    checkpoint = torch.load(policy_path, map_location=torch.device(device))
    validate_low_level_checkpoint(checkpoint, 1, check_classifier=False)
    model.load_state_dict(checkpoint["actor_critic"])
    estimator.load_state_dict(checkpoint["estimator"])
    model.eval()
    estimator.eval()
    return model, estimator


def run_recovery_policy(model, estimator, obs_prop, obs_history, obs_command, obs_latent_zero):
    import torch

    obs_explicit = estimator(obs_prop).reshape(1, -1)
    recovery_info_command = obs_command.clone()
    recovery_info_command[:, 6] = 1
    recovery_command = recovery_info_command[:, :7]
    obs_low_all = torch.cat((obs_prop, obs_explicit, obs_latent_zero, obs_history, recovery_command), dim=-1)
    action = model.act_inference(obs_low_all.detach(), hist_encoding=True)
    return action, obs_explicit, recovery_info_command


def load_low_level_policy(policy_path, mode_count, device, use_half, recover_policy_path=None):
    import torch

    policy_path = require_policy_file(policy_path, "Low-level policy")
    model_ll, estimator_ll = build_low_level_model(mode_count, device, use_half)
    checkpoint = torch.load(policy_path, map_location=torch.device(device))
    validate_low_level_checkpoint(checkpoint, mode_count)
    model_ll.load_state_dict(checkpoint["actor_critic"])
    estimator_ll.load_state_dict(checkpoint["estimator"])
    model_ll.eval()
    estimator_ll.eval()

    obs_latent_zero = torch.zeros((1, 29), device=device, dtype=torch.float16 if use_half else torch.float32)
    recovery_model, recovery_estimator = load_recovery_policy(recover_policy_path, device, use_half)

    def policy(runner, obs, info, delete_memory=False):
        with torch.no_grad():
            obs_prop = obs["obs_main"]
            obs_history = obs["obs_history"].reshape(1, -1)
            obs_command = obs["obs_command"].reshape(1, -1)
            if runner.command_profile.state_estimator.mode == -1 and recovery_model is not None:
                action, obs_explicit, recovery_info_command = run_recovery_policy(
                    recovery_model,
                    recovery_estimator,
                    obs_prop,
                    obs_history,
                    obs_command,
                    obs_latent_zero,
                )
                info["high_command"] = recovery_info_command
                info["obs_explicit"] = obs_explicit
                info["recovery_mode"] = True
                return action, info

            obs_explicit = estimator_ll(obs_prop).reshape(1, -1)
            obs_low_all = torch.cat((obs_prop, obs_explicit, obs_latent_zero, obs_history, obs_command), dim=-1)
            action = model_ll.act_inference(obs_low_all.detach(), hist_encoding=True)
            info["high_command"] = obs_command
            info["obs_explicit"] = obs_explicit
            info["recovery_mode"] = False
            return action, info

    return policy


def load_depth_policy(
    policy_ll_path,
    policy_high_path,
    mode_count,
    cfg_env_high,
    cfg_policy_high,
    device,
    use_half,
    recover_policy_path=None,
    fix_high_estimator=True,
):
    import torch
    import torch.nn.functional as F

    from go2_deploy.module.estimator import Estimator
    from go2_deploy.module.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
    from go2_deploy.module.net_high import Actor, get_activation
    from go2_deploy.utils.command_builder import CommandBuilder

    policy_ll_path = require_policy_file(policy_ll_path, "Low-level policy")
    policy_high_path = require_policy_file(policy_high_path, "High-level policy")

    model_high = Actor(
        num_prop=cfg_env_high.env.n_proprio,
        num_auxiliary=cfg_env_high.env.n_auxiliary,
        num_scan=cfg_env_high.env.n_scan,
        num_priv_latent=cfg_env_high.env.n_priv_latent,
        num_priv_explicit=cfg_env_high.env.n_priv,
        num_hist=cfg_env_high.env.history_len,
        num_actions_d=cfg_env_high.env.num_actions_d,
        num_actions_c=cfg_env_high.env.num_actions_c,
        scan_encoder_dims=cfg_policy_high.policy.scan_encoder_dims,
        actor_hidden_dims=cfg_policy_high.policy.actor_hidden_dims,
        priv_encoder_dims=cfg_policy_high.policy.priv_encoder_dims,
        init_with_orthogonal=cfg_policy_high.policy.init_with_orthogonal,
        tanh_encoder_output=cfg_policy_high.policy.tanh_encoder_output,
        activation=get_activation(cfg_policy_high.policy.activation),
    ).to(device)
    depth_backbone = DepthOnlyFCBackbone58x87(
        cfg_env_high.env.n_proprio,
        cfg_policy_high.policy.scan_encoder_dims[-1],
        cfg_policy_high.depth_encoder.hidden_dims,
    )
    depth_encoder = RecurrentDepthBackbone(depth_backbone, cfg_policy_high.policy.scan_encoder_dims[-1], cfg_env_high).to(device)
    estimator_high = Estimator(
        input_dim=cfg_env_high.env.n_proprio - cfg_env_high.env.n_auxiliary,
        output_dim=cfg_env_high.env.n_priv,
        hidden_dims=cfg_policy_high.estimator.hidden_dims,
    ).to(device)
    model_ll, estimator_ll = build_low_level_model(mode_count, device, use_half)
    recovery_model, recovery_estimator = load_recovery_policy(recover_policy_path, device, use_half)

    high_ckpt = torch.load(policy_high_path, map_location=torch.device(device))
    estimator_high.load_state_dict(high_ckpt["estimator_state_dict"])
    depth_encoder.load_state_dict({k: v for k, v in high_ckpt["depth_encoder_state_dict"].items() if "byol" not in k})
    model_high.load_state_dict(high_ckpt["depth_actor_state_dict"])

    low_ckpt = torch.load(policy_ll_path, map_location=torch.device(device))
    validate_low_level_checkpoint(low_ckpt, mode_count)
    model_ll.load_state_dict(low_ckpt["actor_critic"])
    estimator_ll.load_state_dict(low_ckpt["estimator"])

    for module in (model_high, depth_encoder, estimator_high, model_ll, estimator_ll):
        module.eval()
        if use_half:
            module.half()

    commander = CommandBuilder(mode_count, cfg_env_high, device=device, use_half=use_half)
    dtype = torch.float16 if use_half else torch.float32
    obs_latent_zero = torch.zeros((1, 29), device=device, dtype=dtype)
    obs_proprio_pad = torch.zeros((1, cfg_env_high.env.n_auxiliary), device=device, dtype=dtype)
    scan_info_zero = torch.zeros((1, 132), device=device, dtype=dtype)
    obst_type_start = 59
    obst_type_end = obst_type_start + cfg_env_high.env.n_obst_type

    def policy(runner, obs, info, delete_memory=False):
        with torch.no_grad():
            obs_prop = obs["obs_main"]
            obs_history = obs["obs_history"].reshape(1, -1)
            obs_yaw = obs["obs_yaw"]
            obs_vis = obs["obs_vis"]
            obs_command = obs["obs_command"].reshape(1, -1)

            if delete_memory:
                print("GRU记忆已经清除")
                depth_encoder.delete_memory()

            if runner.command_profile.state_estimator.mode == -1 and recovery_model is not None:
                action, obs_explicit, recovery_info_command = run_recovery_policy(
                    recovery_model,
                    recovery_estimator,
                    obs_prop,
                    obs_history,
                    obs_command,
                    obs_latent_zero,
                )
                info["high_command"] = recovery_info_command
                info["obs_explicit"] = obs_explicit
                info["recovery_mode"] = True
                return action, info

            obs_proprio = torch.cat((obs_prop, obs_proprio_pad), dim=1)
            abs_yaw = (obs_yaw - runner.initial_yaw + torch.pi / 2) % (2 * torch.pi) - torch.pi / 2
            obs_proprio[:, 57] = abs_yaw
            depth_latent = depth_encoder(obs_vis, obs_proprio).reshape(1, -1)
            delta_yaw = 1.5 * depth_latent[:, 32:34]

            runner.abs_yaw.append(abs_yaw.cpu()[0, 0])
            runner.delta_yaw_0.append(delta_yaw.cpu()[0, 0])
            runner.delta_yaw_1.append(delta_yaw.cpu()[0, 1])

            obst_type = F.one_hot(torch.argmax(depth_latent[:, 34:], dim=-1), num_classes=depth_latent[:, 34:].shape[-1])
            obs_proprio[:, 57:59] = delta_yaw
            obs_proprio[:, obst_type_start:obst_type_end] = obst_type

            if fix_high_estimator:
                priv_explicit = estimator_ll(obs_prop).reshape(1, -1)
            else:
                priv_explicit = estimator_high(obs_prop).reshape(1, -1)

            obs_high_all = torch.cat((obs_proprio, scan_info_zero, priv_explicit, obs_history), dim=-1)
            embedding = model_high.forward(obs_high_all, hist_encoding=True, scandots_latent=depth_latent[:, :32])
            actions_d = torch.argmax(torch.softmax(model_high.actor_d(embedding), dim=-1), dim=-1)
            actions_c = model_high.actor_c(embedding)
            ll_command = commander.set_commands(torch.cat([actions_d.unsqueeze(-1), actions_c], dim=-1)).reshape(1, -1)

            obs_explicit = estimator_ll(obs_prop).reshape(1, -1)
            obs_low_all = torch.cat((obs_prop, obs_explicit, obs_latent_zero, obs_history, ll_command), dim=-1)
            action = model_ll.act_inference(obs_low_all, hist_encoding=True)
            info["high_command"] = ll_command
            info["obs_explicit"] = obs_explicit
            info["recovery_mode"] = False
            return action, info

    return policy


def run_deployment(
    experiment_name,
    cfg_ll,
    mode_count,
    policy,
    max_steps,
    log_root,
    device,
    lcm_url,
    max_vel,
    max_yaw_vel,
    use_depth=False,
):
    ensure_device(device)
    import lcm

    from go2_deploy.envs.history_wrapper import HistoryWrapper
    from go2_deploy.envs.lcm_agent import LCMAgent
    from go2_deploy.utils.command_profile import RCControllerProfile
    from go2_deploy.utils.deployment_runner import DeploymentRunner
    from go2_deploy.utils.state_estimator import StateEstimator, StateEstimatorDepth

    lc = lcm.LCM(lcm_url)
    if use_depth:
        state_estimator = StateEstimatorDepth(lc, mode_count, device=device)
    else:
        state_estimator = StateEstimator(lc, mode_count)
    command_profile = RCControllerProfile(
        dt=0.02,
        state_estimator=state_estimator,
        x_scale=max_vel,
        y_scale=1.0,
        yaw_scale=max_yaw_vel,
    )

    hardware_agent = LCMAgent(cfg_ll, state_estimator, command_profile, device=device, lcm_url=lcm_url)
    state_estimator.spin()
    hardware_agent = HistoryWrapper(hardware_agent)
    print("Agent successfully created!")

    runner = DeploymentRunner(experiment_name, state_estimator=state_estimator, log_root=log_root, device=device)
    runner.add_control_agent(hardware_agent, "Ess_infogail_for_3model")
    runner.add_policy(policy)
    runner.add_command_profile(command_profile)
    print("Policy successfully loaded!")
    print(f"max steps {max_steps}")
    runner.run(max_steps=max_steps, logging=True)


def make_parser(description, default_experiment_name):
    parser = argparse.ArgumentParser(description=description)
    add_runtime_args(parser, default_experiment_name)
    return parser
