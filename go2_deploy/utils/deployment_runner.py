import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from go2_deploy.utils.logger import MultiLogger


class DeploymentRunner:
    """Calibration, control loop, logging, and safe pause handling."""

    def __init__(self, experiment_name="unnamed", state_estimator=None, log_root="logs", device="cuda"):
        self.agents = {}
        self.policy = None
        self.command_profile = None
        self.logger = MultiLogger()
        self.se = state_estimator
        self.control_agent_name = None
        self.button_states = np.zeros(4)
        self.is_currently_probing = False
        self.is_currently_logging = [False, False, False, False]
        self.initial_yaw = torch.zeros((1, 1), device=device, dtype=torch.float16)
        self.abs_yaw = []
        self.delta_yaw_0 = []
        self.delta_yaw_1 = []
        self.try_time = 0

        self.log_root = Path(log_root) / experiment_name
        self.init_log_filename()

    def init_log_filename(self):
        stamp = time.strftime("%Y/%m_%d/%H_%M_%S")
        for i in range(100):
            log_dir = self.log_root / f"{stamp}_{i}"
            try:
                log_dir.mkdir(parents=True, exist_ok=False)
                self.log_dirname = log_dir
                self.log_filename = log_dir / "log.pkl"
                return
            except FileExistsError:
                continue
        raise RuntimeError(f"Could not create a unique log directory under {self.log_root}")

    def add_control_agent(self, agent, name):
        self.control_agent_name = name
        self.agents[name] = agent
        self.logger.add_robot(name, agent.cfg)

    def add_policy(self, policy):
        self.policy = policy

    def add_command_profile(self, command_profile):
        self.command_profile = command_profile

    def reset_initial_yaw(self, obs):
        self.initial_yaw = obs["obs_yaw"]

    def calibrate(self, wait=True, low=False, wait_start=True):
        control_obs = None
        for agent_name, agent in self.agents.items():
            if not hasattr(agent, "get_obs"):
                continue
            agent.get_obs()
            joint_pos = agent.dof_pos
            final_goal = np.array([0.0, 1.36, -2.7] * 4) if low else np.zeros(12)
            nominal_joint_pos = agent.default_dof_pos

            if wait:
                print("About to calibrate; the robot will stand [Press R2 to calibrate]")
            else:
                print("About to calibrate; the robot will stand.")
            while wait:
                self.command_profile.get_buttons()
                if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                    print(">>>>>>>>>>>>>>> R2 is pressed <<<<<<<<<<<<<")
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                    break

            cal_action = np.zeros((agent.num_envs, agent.num_actions))
            target = joint_pos - nominal_joint_pos
            target_sequence = []
            while np.max(np.abs(target - final_goal)) > 0.005:
                target -= np.clip(target - final_goal, -0.01, 0.01)
                target_sequence.append(copy.deepcopy(target))

            action_scale = agent.cfg["control"]["action_scale"] if isinstance(agent.cfg, dict) else agent.cfg.control.action_scale
            for target in target_sequence:
                cal_action[:, :12] = target / action_scale
                agent.step(torch.from_numpy(cal_action), start_sig=True)
                agent.get_obs()

            if wait_start:
                print("Starting pose calibrated [Press R2 to start controller]")
                while True:
                    time.sleep(0.01)
                    self.command_profile.get_buttons()
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        print(">>>>>>>>>>>>>>> R2 is pressed again <<<<<<<<<<<<<")
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

            obs = agent.reset()
            if agent_name == self.control_agent_name:
                control_obs = obs
                print("调整结束，进入循环")
        if control_obs is None:
            raise RuntimeError("No control observation was produced during calibration")
        return control_obs

    def run(self, max_steps=10000000, logging=True):
        if self.control_agent_name is None or self.policy is None or self.command_profile is None:
            raise RuntimeError("Runner requires a control agent, policy, and command profile before run().")

        agent_name = self.control_agent_name
        agent = self.agents[agent_name]
        control_obs = agent.reset()

        if hasattr(self.se, "depth_camera"):
            self.se.depth_camera.init_video(self.log_dirname / "depth_video.mp4")

        self.reset_initial_yaw(control_obs)
        self.policy(self, control_obs, {}, True)
        control_obs = self.calibrate(wait=True)
        self.command_profile.state_estimator.right_lower_right_switch_pressed = False

        memory_delete = True
        control_obs = agent.reset()
        self.reset_initial_yaw(control_obs)

        try:
            for i in range(max_steps):
                policy_info = {}
                action, policy_info = self.policy(self, control_obs, policy_info, memory_delete)
                memory_delete = False
                obs, ret, done, info = agent.step(action.detach(), start_sig=False)
                info.update(policy_info)
                info.update(
                    {
                        "observation": obs,
                        "reward": ret,
                        "done": done,
                        "timestep": i,
                        "time": i * agent.dt,
                        "action": action.detach(),
                        "rpy": agent.se.get_rpy(),
                        "torques": agent.torques,
                    }
                )
                if logging:
                    self.logger.log(agent_name, info)
                control_obs = obs
                rpy = agent.se.get_rpy()
                if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                    print("Emergency orientation guard triggered; moving to low pose.")
                    control_obs = self.calibrate(wait=False, low=True)
                    memory_delete = True
                control_obs, memory_delete = self._handle_runtime_controls(agent, agent_name, control_obs, memory_delete)
        except KeyboardInterrupt:
            print("CTRL+C received; shutting down deployment.")
        finally:
            self._shutdown(logging=logging)

    def _handle_runtime_controls(self, agent, agent_name, control_obs, memory_delete):
        self.button_states = self.command_profile.get_buttons()
        se = self.command_profile.state_estimator

        if se.left_lower_left_switch_pressed:
            if not self.is_currently_probing:
                print("START LOGGING")
                self.is_currently_probing = True
                agent.set_probing(True)
                self.init_log_filename()
                self.logger.reset()
            else:
                print("SAVE LOG")
                self.is_currently_probing = False
                agent.set_probing(False)
                control_obs = self.calibrate(wait=False)
                self.logger.save(str(self.log_filename))
                self.init_log_filename()
                self.logger.reset()
                time.sleep(1)
                control_obs = agent.reset()
            se.left_lower_left_switch_pressed = False

        for button in range(4):
            if self.command_profile.currently_triggered[button]:
                if not self.is_currently_logging[button]:
                    print("START LOGGING")
                    self.is_currently_logging[button] = True
                    self.init_log_filename()
                    self.logger.reset()
            else:
                if self.is_currently_logging[button]:
                    print("SAVE LOG")
                    self.is_currently_logging[button] = False
                    control_obs = self.calibrate(wait=False)
                    self.logger.save(str(self.log_filename))
                    self.init_log_filename()
                    self.logger.reset()
                    time.sleep(1)
                    control_obs = agent.reset()

        if se.right_lower_right_switch_pressed:
            control_obs = self.calibrate(wait=False, wait_start=False)
            time.sleep(1)
            se.right_lower_right_switch_pressed = False
            memory_delete = True
            self._save_yaw_plot()
            while not se.right_lower_right_switch_pressed:
                time.sleep(0.01)
            print("重置yaw")
            se.right_lower_right_switch_pressed = False
            control_obs = self.agents[agent_name].reset()
            self.reset_initial_yaw(control_obs)

        return control_obs, memory_delete

    def _save_yaw_plot(self):
        plt.figure()
        plt.plot(range(len(self.abs_yaw)), self.abs_yaw, label="abs_yaw")
        plt.plot(range(len(self.delta_yaw_0)), self.delta_yaw_0, label="delta_yaw_0")
        plt.plot(range(len(self.delta_yaw_1)), self.delta_yaw_1, label="delta_yaw_1")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("value")
        plt.savefig(self.log_dirname / f"yaw_plot_try{self.try_time}.jpg", format="jpg")
        plt.close()
        self.abs_yaw.clear()
        self.delta_yaw_0.clear()
        self.delta_yaw_1.clear()
        self.try_time += 1

    def _shutdown(self, logging=True):
        if hasattr(self.se, "depth_camera"):
            self.se.depth_camera.shutdown()
        if logging:
            self.logger.save(str(self.log_filename))
