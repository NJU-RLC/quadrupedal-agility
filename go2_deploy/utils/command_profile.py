import torch


class RCControllerProfile:
    """Maps wireless-controller state into low-level command vectors."""

    def __init__(self, dt, state_estimator, x_scale=1.0, y_scale=1.0, yaw_scale=1.0, probe_vel_multiplier=1.0):
        self.dt = dt
        self.state_estimator = state_estimator
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.yaw_scale = yaw_scale
        self.probe_vel_multiplier = probe_vel_multiplier
        self.triggered_commands = {i: None for i in range(4)}
        self.currently_triggered = [0, 0, 0, 0]
        self.button_states = [0, 0, 0, 0]

    def get_command(self, t, mode_number, probe=False):
        command = self.state_estimator.get_ess_command().copy()
        command[0] *= self.x_scale
        command[1] *= self.y_scale
        command[2] *= self.yaw_scale
        if probe:
            command[0] *= self.probe_vel_multiplier
            command[2] *= self.probe_vel_multiplier

        reset_timer = False
        prev_button_states = self.button_states[:]
        self.button_states = self.state_estimator.get_buttons()
        for button in range(4):
            profile = self.triggered_commands[button]
            if profile is None:
                continue
            if self.button_states[button] == 1 and prev_button_states[button] == 0:
                if not self.currently_triggered[button]:
                    profile.reset(t)
                    reset_timer = True
                    self.currently_triggered[button] = True
                else:
                    self.currently_triggered[button] = False
            if self.currently_triggered[button] and t < profile.max_timestep:
                command = profile.get_command(t)
        return command, reset_timer

    def add_triggered_command(self, button_idx, command_profile):
        self.triggered_commands[button_idx] = command_profile

    def get_buttons(self):
        return self.state_estimator.get_buttons()


class ConstantCommandProfile:
    def __init__(self, dt, command, max_time_s=10.0):
        self.dt = dt
        self.max_timestep = int(max_time_s / dt)
        self.command = torch.as_tensor(command)
        self.start_time = 0.0

    def reset(self, reset_time):
        self.start_time = reset_time

    def get_command(self, t):
        return self.command
