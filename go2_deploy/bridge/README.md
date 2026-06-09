# Unitree SDK2 Bridge

`go2_deploy` 的 Python 策略不直接调用 Unitree SDK2，而是通过 LCM 与 C++ bridge 通信。

bridge 负责：

- 从 Unitree SDK2 读取 Go2 低层状态、IMU、足端力和遥控器输入
- 通过 LCM 发布机器人状态
- 订阅 Python 策略发出的关节目标
- 把关节目标转成 Unitree low-level command

## 推荐方式：使用 Docker 内置 bridge

推荐使用这个 Docker 环境：

```text
https://github.com/BlackVegetab1e/Leggedgym-Deploy-Docker
```

该镜像已经内置并编译好：

- LCM 1.5.1
- Unitree SDK2
- `lcm_position_go2`

进入容器后直接运行：

```bash
cd /workspace/unitree_sdk2_bin/build
./lcm_position_go2 eth0
```

把 `eth0` 替换成连接 Go2 的网卡名。

## 非 Docker 方式

如果不用 Docker，需要自己准备 Unitree SDK2、LCM 和 bridge 源码。`go2_deploy` 不再内置这些 C++/SDK 文件。

```text
unitree_sdk2_bin/lcm_position_go2.cpp
unitree_sdk2_bin/CMakeLists.txt
```

编译：

```bash
cd /path/to/unitree_sdk2_bin
mkdir -p build
cd build
cmake ..
make -j
```

运行：

```bash
./lcm_position_go2 eth0
```

## 与 Python 策略配合

保持 bridge 终端打开，然后在另一个终端运行：

```bash
python3 -m go2_deploy.scripts.deploy_policy_with_controller_3
```

bridge 和 Python 策略之间使用以下 LCM topic：

- Python 订阅：
  - `leg_control_data`
  - `state_estimator_data`
  - `rc_command`
- Python 发布：
  - `pd_plustau_targets`

## 说明

`go2_deploy` 已经内置 Python 侧运行需要的 LCM message 类型，但不包含 LCM 源码、Unitree SDK2 或 C++ bridge。推荐直接使用 Docker 镜像里的 `/workspace/unitree_sdk2_bin/build/lcm_position_go2`。
