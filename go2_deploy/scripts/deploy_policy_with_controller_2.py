from pathlib import Path

from go2_deploy.config.go2_ll_config_0515 import Go2CanterCfg
from go2_deploy.scripts.common import (
    add_low_level_args,
    ensure_device,
    load_low_level_policy,
    make_parser,
    run_deployment,
)


MODE_NUMBER = 2
DEFAULT_LL_POLICY = Path("go2_deploy/models/ll_policy/2_model/model.pt")


def parse_args():
    parser = make_parser("Deploy the 2-mode low-level controller on Unitree Go2.", "go2_controller_2")
    add_low_level_args(parser, DEFAULT_LL_POLICY)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_device(args.device)
    use_half = not args.float32
    policy = load_low_level_policy(args.ll_policy, MODE_NUMBER, args.device, use_half, args.recover_policy)
    print("now low model", args.ll_policy)
    run_deployment(
        experiment_name=args.experiment_name,
        cfg_ll=Go2CanterCfg(),
        mode_count=MODE_NUMBER,
        policy=policy,
        max_steps=args.max_steps,
        log_root=args.log_root,
        device=args.device,
        lcm_url=args.lcm_url,
        max_vel=args.max_vel,
        max_yaw_vel=args.max_yaw_vel,
        use_depth=False,
    )


if __name__ == "__main__":
    main()
