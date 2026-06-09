import torch


class CommandBuilder:
    """Converts high-level discrete/continuous actions into low-level commands."""

    def __init__(self, mode_count, cfg, device="cuda", use_half=True):
        if mode_count not in (2, 3):
            raise ValueError("Depth high-level command builder currently supports 2 or 3 modes.")
        self.mode_count = mode_count
        self.device = torch.device(device)
        self.dtype = torch.float16 if use_half else torch.float32

        lin_vel_x = torch.tensor(cfg.commands.max_ranges.lin_vel_x, device=self.device, dtype=self.dtype)
        lin_vel_y = torch.tensor(cfg.commands.max_ranges.lin_vel_y, device=self.device, dtype=self.dtype)
        ang_vel_yaw = torch.tensor(cfg.commands.max_ranges.ang_vel_yaw, device=self.device, dtype=self.dtype)
        jump_height = torch.tensor(cfg.commands.max_ranges.jump_height, device=self.device, dtype=self.dtype)
        locomotion_height = torch.tensor(cfg.commands.max_ranges.locomotion_height, device=self.device, dtype=self.dtype)

        low = torch.stack(
            [
                lin_vel_x[:, 0],
                lin_vel_y[:, 0],
                ang_vel_yaw[:, 0],
                jump_height[0].repeat(mode_count),
                locomotion_height[0].repeat(mode_count),
            ],
            dim=-1,
        )
        high = torch.stack(
            [
                lin_vel_x[:, 1],
                lin_vel_y[:, 1],
                ang_vel_yaw[:, 1],
                jump_height[1].repeat(mode_count),
                locomotion_height[1].repeat(mode_count),
            ],
            dim=-1,
        )
        self.command_offset_mat = low
        self.command_range_mat = high - low
        self.jump_mode = torch.tensor(2 if mode_count == 3 else 1, device=self.device)
        self.num_actions_c = torch.tensor([cfg.env.num_actions_c], dtype=torch.long, device=self.device)
        self.action_arrange = torch.arange(cfg.env.num_actions_c, device=self.device) + 1
        self.latent_c_one_hot = torch.eye(mode_count, device=self.device, dtype=self.dtype)
        self.latent_eps = torch.ones((1, 1), device=self.device, dtype=self.dtype)

    def set_commands(self, actions):
        action_d = actions[0, 0].unsqueeze(0).long().clamp(0, self.mode_count - 1)
        indices = action_d * self.num_actions_c + self.action_arrange
        commands = torch.clip(actions[0, indices], -1, 1)
        self.latent_eps[0, 0] = commands[-1].clone()
        latent_c = self.latent_c_one_hot[action_d]
        commands01 = (commands + 1) / 2
        commands_scaled = self.command_offset_mat[action_d] + self.command_range_mat[action_d] * commands01[:5]
        if int(action_d.item()) == int(self.jump_mode.item()):
            commands_scaled[0, 4] = 0
        else:
            commands_scaled[0, 3] = 0
        return torch.cat([commands_scaled, self.latent_eps, latent_c], dim=-1)
