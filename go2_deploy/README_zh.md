# Go2 Deploy

`go2_deploy` 是 ESS-InfoGAIL 在 Unitree Go2 上的部署代码。它包含 Python 侧部署所需的策略网络、配置文件、LCM types、深度图 `.so` 处理扩展和分任务部署入口。

Unitree SDK2、LCM 编译环境和 C++ bridge 不放在本目录中，推荐直接使用 Docker 环境里已经编译好的版本。

## 部署原理

部署流程由三部分组成：

1. Python policy 脚本基于 PyTorch 进行策略推理。
2. LCM 作为通信中间层，连接 Python policy 和 Unitree SDK2 bridge。
3. `lcm_position_go2` 通过 Unitree SDK2 与机器狗通信：从机器人读取观测并通过 LCM 发布，同时从 LCM 接收 policy 输出的 action 并发送给机器人。

`lcm_position_go2` 的编译依赖较多，建议直接使用下面的 Docker 环境。`lcm_position_go2` 和 Python policy 通过 LCM 通信，本质上是网络通信，因此在同一网络环境下即可互相传输数据。

信息流如下：

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

其中 Python policy 订阅机器人观测和遥控器状态，推理得到关节目标后发布到 `pd_plustau_targets`；`lcm_position_go2` 再把该消息转换为 Unitree low-level command。

## Docker 环境

推荐使用这个部署环境：

```text
https://github.com/BlackVegetab1e/Leggedgym-Deploy-Docker
```

该镜像内置 PyTorch、LCM 1.5.1、Unitree SDK2 和已经编译好的 `lcm_position_go2`。

构建镜像：

```bash
git clone https://github.com/BlackVegetab1e/Leggedgym-Deploy-Docker.git
cd Leggedgym-Deploy-Docker
docker build -t go2_env:r35.3.1 .
```

创建容器：

```bash
docker run \
  --name go2_deploy \
  --runtime nvidia \
  --network host \
  --ipc=host \
  -v /path/to/quadrupedal-agility:/workspace/policy_code \
  -it go2_env:r35.3.1 bash
```

之后再次进入：

```bash
docker start go2_deploy
docker exec -it go2_deploy bash
cd /workspace/policy_code
```

如果 Python 找不到当前工程：

```bash
export PYTHONPATH=/workspace/policy_code:$PYTHONPATH
```

## 功能

- 低层步态策略部署：2/3/5 步态分别有独立入口
- 深度/高层策略部署：普通 3 步态、hurdle 分别有独立入口；box 入口已预留，但当前没有随包提供模型
- 遥控器控制速度、yaw 和步态切换
- 启动前姿态校准，按 `R2` 确认
- 运行日志、深度视频和 yaw 曲线保存

## 模型

当前默认模型路径如下：

| 入口 | 默认模型 |
| --- | --- |
| 3 模态低层 | `go2_deploy/models/ll_policy/3_model/model.pt` |
| 2 模态低层 | `go2_deploy/models/ll_policy/2_model/model.pt` |
| 5 模态低层 | `go2_deploy/models/ll_policy/5_model/model.pt` |
| cruft 深度 low | `go2_deploy/models/ll_policy/3_model/model.pt` |
| cruft 深度 high | `go2_deploy/models/high_policy/cruft/model.pt` |
| hurdle 深度 low | `go2_deploy/models/ll_policy/hurdle/model.pt` |
| hurdle 深度 high | `go2_deploy/models/high_policy/hurdle/model.pt` |
| recover | `go2_deploy/models/recover_policy/1_model/model.pt` |

box 深度策略目前没有看到对应模型，已预留位置：

```text
go2_deploy/models/ll_policy/box/model.pt
go2_deploy/models/high_policy/box/model.pt
```

低层 checkpoint 需要包含：

```text
actor_critic
estimator
```

高层 checkpoint 需要包含：

```text
estimator_state_dict
depth_encoder_state_dict
depth_actor_state_dict
```

## 运行

先启动 SDK-LCM bridge：

```bash
cd /workspace/unitree_sdk2_bin/build
./lcm_position_go2 eth0
```

把 `eth0` 换成连接 Go2 的网卡名，可以用 `ip addr` 查看。

再开另一个终端运行 Python 策略：

```bash
cd /workspace/policy_code
```

低层 3 模态策略：

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_controller_3
```

短测试：

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_controller_3 500
```

如果模型不在默认路径：

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_controller_3 \
  --ll-policy /path/to/model.pt
```

其他低层入口：

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_controller_2
python3 -m go2_deploy.scripts.deploy_policy_with_controller_5
```

深度/高层策略示例：

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_depth_cruft
```

其他深度入口：

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_depth_hurdle
```

box 深度策略目前缺少默认模型，不建议直接运行 `deploy_policy_with_depth_box`。等补齐 `ll_policy/box/model.pt` 和 `high_policy/box/model.pt` 后再启用。

## 遥控器

- `R2`：确认校准、暂停、继续
- `L2`：开始/停止日志记录
- `Start`：进入恢复模式。若存在 `go2_deploy/models/recover_policy/1_model/model.pt`，会切到恢复模型；否则使用低速手动恢复 command。
- `A/B/X/Y/R1`：切换步态或模式
- 左摇杆：前进和侧向速度
- 右摇杆左右：yaw 速度

## 日志

日志保存在：

```text
go2_deploy/logs/<experiment_name>/<date>/<time>_<index>/
```

可能包含：

```text
log.pkl
depth_video.mp4
yaw_plot_try*.jpg
```

## 安全

真实机器人运行前请确认：

1. 第一次测试必须吊起机器人。
2. `lcm_position_go2` 已收到 low-level state。
3. `sport_mode` 已关闭。
4. 模型和入口脚本对应，例如 3 步态模型只能跑 `deploy_policy_with_controller_3.py` 或 3 步态顶层入口。
5. Docker 使用 `--network host`。
6. 物理急停可用。
