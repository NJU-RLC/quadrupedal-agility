# Go2 Deploy

[中文文档](README_zh.md)

`go2_deploy` contains the Unitree Go2 deployment code for the quadrupedal-agility policies. It includes the Python policy networks, deployment configs, LCM message types, depth image processing extension, pretrained deployment checkpoints, and task-specific entry points.

Unitree SDK2, the LCM build environment, and the C++ bridge are not included in this directory. The recommended path is to use the Docker environment that already contains the compiled bridge.

## Deployment Architecture

The deployment pipeline has three parts:

1. The Python policy scripts run PyTorch inference.
2. LCM connects the Python policy process and the Unitree SDK2 bridge.
3. `lcm_position_go2` communicates with the Go2 through Unitree SDK2. It reads robot observations and publishes them through LCM, then receives policy actions through LCM and sends them to the robot.

`lcm_position_go2` has several build dependencies, so using the Docker environment below is recommended. The bridge and the Python policy communicate over LCM and only need to be on the same network.

```text
                Unitree Go2 Robot
        ┌─────────────────────────────┐
        │  LowState / IMU / Joystick  │
        │  LowCmd                     │
        └──────────────┬──────────────┘
                       │ Unitree SDK2
                       │
              ┌────────▼────────┐
              │ lcm_position_go2│
              │   C++ bridge    │
              └───────┬─────────┘
                      │
      obs topics      │ LCM                      action topic
 ┌────────────────────▼────────────────────────────────────┐
 │ leg_control_data / state_estimator_data / rc_command    │
 │ pd_plustau_targets                                      │
 └────────────────────┬────────────────────────────────────┘
                      │
              ┌───────▼────────┐
              │ Python policy  │
              │  go2_deploy    │
              └────────────────┘
```

The Python policy subscribes to robot observations and remote controller state, runs policy inference, and publishes joint targets to `pd_plustau_targets`. `lcm_position_go2` converts that message into a Unitree low-level command.

## Docker Environment

Recommended deployment environment:

```text
https://github.com/BlackVegetab1e/Leggedgym-Deploy-Docker
```

The image includes PyTorch, LCM 1.5.1, Unitree SDK2, and a compiled `lcm_position_go2`.

Build the image:

```bash
git clone https://github.com/BlackVegetab1e/Leggedgym-Deploy-Docker.git
cd Leggedgym-Deploy-Docker
docker build -t go2_env:r35.3.1 .
```

Create the container:

```bash
docker run \
  --name go2_deploy \
  --runtime nvidia \
  --network host \
  --ipc=host \
  -v /path/to/quadrupedal-agility:/workspace/policy_code \
  -it go2_env:r35.3.1 bash
```

Re-enter the container later:

```bash
docker start go2_deploy
docker exec -it go2_deploy bash
cd /workspace/policy_code
```

If Python cannot find the project package:

```bash
export PYTHONPATH=/workspace/policy_code:$PYTHONPATH
```

## Features

- Low-level gait policy deployment with separate entry points for 2-mode, 3-mode, and 5-mode policies.
- Depth/high-level policy deployment for the regular 3-mode task and the hurdle task.
- A reserved box-depth entry point. The matching box models are not included yet.
- Remote controller control for velocity, yaw, and mode switching.
- Startup pose calibration confirmed with `R2`.
- Runtime logging, depth video recording, and yaw curve export.

## Models

Default model paths:

| Entry point | Default model |
| --- | --- |
| 3-mode low-level | `go2_deploy/models/ll_policy/3_model/model.pt` |
| 2-mode low-level | `go2_deploy/models/ll_policy/2_model/model.pt` |
| 5-mode low-level | `go2_deploy/models/ll_policy/5_model/model.pt` |
| cruft depth low | `go2_deploy/models/ll_policy/3_model/model.pt` |
| cruft depth high | `go2_deploy/models/high_policy/cruft/model.pt` |
| hurdle depth low | `go2_deploy/models/ll_policy/hurdle/model.pt` |
| hurdle depth high | `go2_deploy/models/high_policy/hurdle/model.pt` |
| recovery | `go2_deploy/models/recover_policy/1_model/model.pt` |

The box-depth policy is reserved but the matching models are not included:

```text
go2_deploy/models/ll_policy/box/model.pt
go2_deploy/models/high_policy/box/model.pt
```

Low-level checkpoints must contain:

```text
actor_critic
estimator
```

High-level checkpoints must contain:

```text
estimator_state_dict
depth_encoder_state_dict
depth_actor_state_dict
```

## Running

Start the SDK-LCM bridge first:

```bash
cd /workspace/unitree_sdk2_bin/build
./lcm_position_go2 eth0
```

Replace `eth0` with the network interface connected to the Go2. Use `ip addr` to inspect available interfaces.

Then open another terminal for the Python policy:

```bash
cd /workspace/policy_code
```

Run the low-level 3-mode policy:

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_controller_3
```

Short test:

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_controller_3 500
```

Use a custom model path:

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_controller_3 \
  --ll-policy /path/to/model.pt
```

Other low-level entry points:

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_controller_2
python3 -m go2_deploy.scripts.deploy_policy_with_controller_5
```

Run the depth/high-level policy:

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_depth_cruft
```

Other depth entry points:

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_depth_hurdle
```

The box-depth policy does not have default models yet. Do not run `deploy_policy_with_depth_box` until `ll_policy/box/model.pt` and `high_policy/box/model.pt` are provided.

## Remote Controller

- `R2`: confirm calibration, pause, and resume.
- `L2`: start or stop logging.
- `Start`: enter recovery mode. If `go2_deploy/models/recover_policy/1_model/model.pt` exists, the recovery model is used; otherwise a low-speed manual recovery command is used.
- `A/B/X/Y/R1`: switch gait or mode.
- Left stick: forward and lateral velocity.
- Right stick left/right: yaw velocity.

## Logs

Logs are saved under:

```text
go2_deploy/logs/<experiment_name>/<date>/<time>_<index>/
```

Possible outputs:

```text
log.pkl
depth_video.mp4
yaw_plot_try*.jpg
```

## Safety

Before running on the real robot:

1. Suspend the robot for the first test.
2. Confirm that `lcm_position_go2` receives low-level state.
3. Make sure `sport_mode` is disabled.
4. Match the model and entry point. For example, a 3-mode model should only run with `deploy_policy_with_controller_3.py` or the matching 3-mode high-level entry point.
5. Run Docker with `--network host`.
6. Make sure a physical emergency stop is available.
