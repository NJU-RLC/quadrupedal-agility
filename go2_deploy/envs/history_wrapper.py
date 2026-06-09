import torch


class HistoryWrapper:
    """Adds fixed-length proprioceptive history to a hardware agent."""

    def __init__(self, env):
        self.env = env
        cfg = env.cfg
        self.obs_history_length = cfg["env"]["history_len"] if isinstance(cfg, dict) else cfg.env.history_len
        self.num_obs_history = self.obs_history_length * env.num_prop
        self.last_obs_main = torch.zeros(1, env.num_prop, device=env.device, dtype=torch.float16)
        self.obs_history = torch.zeros(
            env.num_envs,
            self.num_obs_history,
            dtype=torch.float16,
            device=env.device,
            requires_grad=False,
        )

    def get_obs(self):
        obs_part = self.env.get_obs()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_prop :], self.last_obs_main), dim=-1)
        self.last_obs_main = obs_part["obs_main"]
        return self._merge(obs_part)

    def step(self, action, start_sig=False):
        obs, rew, done, info = self.env.step(action, start_sig=start_sig)
        return self._merge(obs), rew, done, info

    def reset(self):
        obs_part = self.env.reset()
        self.obs_history.zero_()
        self.last_obs_main = obs_part["obs_main"]
        return self._merge(obs_part)

    def _merge(self, obs_part):
        return {
            "obs_main": obs_part["obs_main"],
            "obs_yaw": obs_part["obs_yaw"],
            "obs_vis": obs_part["obs_vis"],
            "obs_command": obs_part["obs_command"],
            "obs_history": self.obs_history,
        }

    def __getattr__(self, name):
        return getattr(self.env, name)
