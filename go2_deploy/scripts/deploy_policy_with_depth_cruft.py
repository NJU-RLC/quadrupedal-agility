from pathlib import Path

from go2_deploy.config.go2_ll_config_0515 import Go2CanterCfg
from go2_deploy.scripts.common import (
    add_depth_args,
    ensure_device,
    load_depth_policy,
    make_parser,
    run_deployment,
)


MODE_NUMBER = 3
DEFAULT_LL_POLICY = Path("go2_deploy/models/ll_policy/3_model/model.pt")
DEFAULT_HIGH_POLICY = Path("go2_deploy/models/high_policy/cruft/model.pt")


def parse_args():
    parser = make_parser("Deploy the 3-mode depth high-level policy on Unitree Go2.", "go2_depth_cruft")
    add_depth_args(parser, DEFAULT_LL_POLICY, DEFAULT_HIGH_POLICY)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_device(args.device)
    from go2_deploy.config.go2_parkour_config import Go2ParkourCfg, Go2ParkourCfgPPO

    use_half = not args.float32
    cfg_env_high = Go2ParkourCfg()
    cfg_policy_high = Go2ParkourCfgPPO()
    policy = load_depth_policy(
        policy_ll_path=args.ll_policy,
        policy_high_path=args.high_policy,
        mode_count=MODE_NUMBER,
        cfg_env_high=cfg_env_high,
        cfg_policy_high=cfg_policy_high,
        device=args.device,
        use_half=use_half,
        recover_policy_path=args.recover_policy,
        fix_high_estimator=args.fix_high_estimator,
    )
    print("now low model", args.ll_policy)
    print("now high model", args.high_policy)
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
        use_depth=True,
    )


if __name__ == "__main__":
    main()
