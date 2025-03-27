import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

from legged_gym.envs.base import observation_buffer


# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_obs
        self.num_obs_disc = cfg.env.num_obs_disc
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.include_history_steps = cfg.env.include_history_steps

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        if cfg.env.include_history_steps is not None:
            self.obs_buf_history = observation_buffer.ObservationBuffer(
                self.num_envs, self.num_obs,
                self.include_history_steps, self.device)
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.obs_disc_buf = torch.zeros(self.num_envs, self.num_obs_disc, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)  # Prepares simulation with buffer allocations

        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_W, "forward")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_S, "backward")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_A, "turn_left")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_D, "turn_right")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_T, "stop")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_H, "loco_height_up")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_L, "loco_height_down")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_SPACE, "jump")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_1, "mode_1")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_2, "mode_2")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_3, "mode_3")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_4, "mode_4")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_5, "mode_5")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT, "epsilon_down")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_RIGHT, "epsilon_up")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_UP, "jump_height_up")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_DOWN, "jump_height_down")

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "forward" and evt.value > 0:
                    self.commands[:, 0] = self.commands[0, 0] + 0.05
                    self.commands[:, 1] = 0
                    self.commands[:, 2] = 0
                elif evt.action == "backward" and evt.value > 0:
                    self.commands[:, 0] = self.commands[0, 0] - 0.05
                    self.commands[:, 1] = 0
                    self.commands[:, 2] = 0
                elif evt.action == "turn_left" and evt.value > 0:
                    self.commands[:, 1] = 0.3
                    self.commands[:, 2] = 0.6
                elif evt.action == "turn_right" and evt.value > 0:
                    self.commands[:, 1] = -0.3
                    self.commands[:, 2] = -0.6
                elif evt.action == "stop" and evt.value > 0:
                    self.commands[:, 0] = 0.0
                    self.commands[:, 1] = 0.0
                    self.commands[:, 2] = 0.0

                elif evt.action == "jump" and evt.value > 0:
                    print("jump!")
                elif evt.action == "epsilon_down" and evt.value > 0:
                    self.latent_eps[:, :] = torch.clip(self.latent_eps[:, :] - 0.1, -1, 1)
                    print("epsilon_down: ", self.latent_eps.detach().cpu().numpy()[0, 0])
                elif evt.action == "epsilon_up" and evt.value > 0:
                    self.latent_eps[:, :] = torch.clip(self.latent_eps[:, :] + 0.1, -1, 1)
                    print("epsilon_up: ", self.latent_eps.detach().cpu().numpy()[0, 0])

                for i in range(self.latent_c.shape[-1]):
                    if evt.action == "mode_{}".format(i+1) and evt.value > 0:
                        self.latent_c[:, :] = 0
                        self.latent_c[:, i] = 1
                        self.commands[:, 0] = 2.5
                        self.commands[:, 3] = 0.0
                        self.latent_eps[:, :] = 0.0
                        if i == (self.latent_c.shape[-1] - 1):
                            self.commands[:, 3] = 0.55
                        break

                if evt.action == "loco_height_up" and evt.value > 0:
                    self.commands[:, 4] = torch.clip(self.commands[:, 4] + 0.01,
                                                     self.command_ranges["locomotion_height"][0],
                                                     self.command_ranges["locomotion_height"][1])
                    print("loco_height_up: ", self.commands[:, 4].detach().cpu().numpy()[0])
                elif evt.action == "loco_height_down" and evt.value > 0:
                    self.commands[:, 4] = torch.clip(self.commands[:, 4] - 0.01,
                                                     self.command_ranges["locomotion_height"][0],
                                                     self.command_ranges["locomotion_height"][1])
                    print("loco_height_down: ", self.commands[:, 4].detach().cpu().numpy()[0])

                if evt.action == "jump_height_up" and evt.value > 0:
                    self.commands[:, 3] = torch.clip(self.commands[:, 3] + 0.01,
                                                     self.command_ranges["jump_height"][0],
                                                     self.command_ranges["jump_height"][1])
                    print("jump_height_up: ", self.commands[:, 3].detach().cpu().numpy()[0])
                elif evt.action == "jump_height_down" and evt.value > 0:
                    self.commands[:, 3] = torch.clip(self.commands[:, 3] - 0.01,
                                                     self.command_ranges["jump_height"][0],
                                                     self.command_ranges["jump_height"][1])
                    print("jump_height_down: ", self.commands[:, 3].detach().cpu().numpy()[0])

                if evt.action != "QUIT" and evt.action != "toggle_viewer_sync" and evt.value > 0:
                    print("command (x, y, yaw): {}, {}, {}".format(self.commands[0, 0], self.commands[0, 1],
                                                                   self.commands[0, 2]))

                if self.cfg.env.play_mode:
                    # clip vel according to latent_c
                    if self.num_mocap == 1:
                        latent_c_idx = 0
                    else:
                        latent_c_idx = torch.argmax(self.latent_c[0, :])
                    lin_vel_x_range = self.command_ranges["lin_vel_x"][latent_c_idx]
                    lin_vel_y_range = self.command_ranges["lin_vel_y"][latent_c_idx]
                    ang_vel_yaw_range = self.command_ranges["ang_vel_yaw"][latent_c_idx]
                    self.commands[:, 0] = torch.clip(self.commands[:, 0], lin_vel_x_range[0], lin_vel_x_range[1])
                    self.commands[:, 1] = torch.clip(self.commands[:, 1], lin_vel_y_range[0], lin_vel_y_range[1])
                    self.commands[:, 2] = torch.clip(self.commands[:, 2], ang_vel_yaw_range[0], ang_vel_yaw_range[1])

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
