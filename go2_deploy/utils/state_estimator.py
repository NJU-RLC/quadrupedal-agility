import select
import threading
import time

import numpy as np

from go2_deploy.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from go2_deploy.lcm_types.rc_command_lcmt import rc_command_lcmt
from go2_deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt


def get_rotation_matrix_from_rpy(rpy):
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


class StateEstimator:
    """LCM subscriber for Go2 proprioception and wireless controller state."""

    def __init__(self, lc, mode_number):
        self.lc = lc
        self.mode_number = mode_number
        self.joint_idxs = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.contact_idxs = [1, 0, 3, 2]

        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.tau_est = np.zeros(12)
        self.euler = np.zeros(3)
        self.R = np.eye(3)
        self.body_lin_vel = np.zeros(3)
        self.body_ang_vel = np.zeros(3)
        self.world_lin_vel = np.zeros(3)
        self.world_ang_vel = np.zeros(3)
        self.body_loc = np.zeros(3)
        self.body_quat = np.array([0, 0, 0, 1])
        self.contact_state = np.ones(4)

        self.mode = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        self.right_lower_left_switch = 0
        self.right_lower_right_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        self.right_lower_left_switch_pressed = 0
        self.right_lower_right_switch_pressed = 0

        self.use_cameras = False
        self.init_time = time.time()
        self.received_first_legdata = False

        self.imu_subscription = self.lc.subscribe("state_estimator_data", self._imu_cb)
        self.legdata_state_subscription = self.lc.subscribe("leg_control_data", self._legdata_cb)
        self.rc_command_subscription = self.lc.subscribe("rc_command", self._rc_command_cb)
        print(f"Deployment mode count: {mode_number}")

    def get_latent(self):
        if self.mode_number not in (1, 2, 3, 5):
            raise ValueError("mode_number must be one of 1, 2, 3, 5")
        mode = self.mode if 0 <= self.mode < self.mode_number else 0
        latent = np.zeros(self.mode_number)
        latent[mode] = 1
        return latent

    def get_body_linear_vel(self):
        return np.array(self.body_lin_vel)

    def get_body_angular_vel(self):
        return np.array(self.body_ang_vel)

    def get_contact_state(self):
        return (self.contact_state[self.contact_idxs] > 14) - 0.5

    def get_rpy(self):
        return self.euler

    def get_latent_eps(self):
        return 0

    def get_dof_pos(self):
        return self.joint_pos[self.joint_idxs]

    def get_dof_vel(self):
        return self.joint_vel[self.joint_idxs]

    def get_tau_est(self):
        return self.tau_est[self.joint_idxs]

    def get_buttons(self):
        return np.array(
            [
                self.left_lower_left_switch,
                self.left_upper_switch,
                self.right_lower_right_switch,
                self.right_upper_switch,
            ]
        )

    def get_ess_command(self):
        mode_number = self.mode_number
        x = float(self.left_stick[1])
        y = float(self.left_stick[0])
        yaw = float(self.right_stick[0])
        if self.mode == -1:
            return np.array([x * 1.5, y * -0.3, yaw * -1.57, 0.0, 0.25])

        mode = self.mode if 0 <= self.mode < mode_number else 0

        table = {
            1: [(1.5, -0.3, -1.57, 0.0, 0.25)],
            2: [(2.5, -0.3, -0.5, 0.0, 0.30), (2.5, -0.3, -0.5, 0.55, 0.0)],
            3: [(1.8, -0.3, -1.57, 0.0, 0.25), (2.5, -0.3, -0.5, 0.0, 0.25), (2.0, -0.3, -0.5, 0.45, 0.0)],
            5: [
                (0.6, 0.15, -1.0, 0.0, 0.25),
                (1.5, 0.30, -1.57, 0.0, 0.25),
                (1.8, -0.30, -1.57, 0.0, 0.25),
                (2.5, -0.30, -0.5, 0.0, 0.25),
                (2.0, -0.30, -0.5, 0.45, 0.0),
            ],
        }
        x_scale, y_scale, yaw_scale, jump_height, locomotion_height = table[mode_number][mode]
        x_offset = 1.0 if mode_number == 2 else 0.0
        return np.array([x * x_scale + x_offset, y * y_scale, yaw * yaw_scale, jump_height, locomotion_height])

    def get_ess_with1model_command(self):
        return self.get_ess_command()

    def get_ess_with2model_command(self):
        return self.get_ess_command()

    def get_ess_with3model_command(self):
        return self.get_ess_command()

    def get_ess_with5model_command(self):
        return self.get_ess_command()

    def _legdata_cb(self, channel, data):
        if not self.received_first_legdata:
            self.received_first_legdata = True
            print(f"Received first low-level data after {time.time() - self.init_time:.2f}s")
        msg = leg_control_data_lcmt.decode(data)
        self.joint_pos = np.array(msg.q)
        self.joint_vel = np.array(msg.qd)
        self.tau_est = np.array(msg.tau_est)

    def _imu_cb(self, channel, data):
        msg = state_estimator_lcmt.decode(data)
        self.body_quat = msg.quat
        self.body_loc = msg.p
        self.body_lin_vel = msg.vBody
        self.body_ang_vel = msg.omegaBody
        self.world_lin_vel = msg.vWorld
        self.world_ang_vel = msg.omegaWorld
        self.euler = np.array(msg.rpy)
        self.R = get_rotation_matrix_from_rpy(self.euler)
        self.contact_state = np.array(msg.contact_estimate)

    def _rc_command_cb(self, channel, data):
        msg = rc_command_lcmt.decode(data)
        self.left_upper_switch_pressed = (msg.left_upper_switch and not self.left_upper_switch) or self.left_upper_switch_pressed
        self.left_lower_left_switch_pressed = (msg.left_lower_left_switch and not self.left_lower_left_switch) or self.left_lower_left_switch_pressed
        self.left_lower_right_switch_pressed = (msg.left_lower_right_switch and not self.left_lower_right_switch) or self.left_lower_right_switch_pressed
        self.right_upper_switch_pressed = (msg.right_upper_switch and not self.right_upper_switch) or self.right_upper_switch_pressed
        self.right_lower_left_switch_pressed = (msg.right_lower_left_switch and not self.right_lower_left_switch) or self.right_lower_left_switch_pressed
        self.right_lower_right_switch_pressed = (msg.right_lower_right_switch and not self.right_lower_right_switch) or self.right_lower_right_switch_pressed
        self.mode = msg.mode
        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.left_upper_switch = msg.left_upper_switch
        self.left_lower_left_switch = msg.left_lower_left_switch
        self.left_lower_right_switch = msg.left_lower_right_switch
        self.right_upper_switch = msg.right_upper_switch
        self.right_lower_left_switch = msg.right_lower_left_switch
        self.right_lower_right_switch = msg.right_lower_right_switch

    def poll(self):
        while True:
            rfds, _, _ = select.select([self.lc.fileno()], [], [], 0.01)
            if rfds:
                self.lc.handle()

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=True)
        self.run_thread.start()

    def close(self):
        self.lc.unsubscribe(self.legdata_state_subscription)
        self.lc.unsubscribe(self.imu_subscription)
        self.lc.unsubscribe(self.rc_command_subscription)


class StateEstimatorDepth(StateEstimator):
    def __init__(self, lc, mode_number, device="cuda"):
        super().__init__(lc, mode_number=mode_number)
        from go2_deploy.utils.depth_camera import DepthCamera

        self.depth_camera = DepthCamera(device=device)
        self.use_cameras = True

    def get_depth_image(self):
        return self.depth_camera.depth_torch

    def get_depth_time(self):
        return self.depth_camera.camera_time
