import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd
import torch.nn.functional as F

from rsl_rl.utils import utils

DISC_LOGIT_INIT_SCALE = 1.0


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_disc_obs, dim_c, dt, disc_loss_function, reward_i_normalizer, reward_i_coef,
                 reward_us_coef, reward_ss_coef, reward_t_coef, disc_obs_len, hidden_units, normalizer, device):
        super(Discriminator, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.num_disc_obs = num_disc_obs
        self.dim_c = dim_c
        self.dt = dt
        self.disc_loss_function = disc_loss_function
        self.reward_i_normalizer = reward_i_normalizer

        self.disc_obs_len = disc_obs_len

        self.reward_i_coef = reward_i_coef
        self.reward_us_coef = reward_us_coef
        self.reward_ss_coef = reward_ss_coef
        self.reward_t_coef = reward_t_coef

        self.normalizer = normalizer

        layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_units[-1], 1)
        self.classifier = nn.Linear(hidden_units[-1], dim_c)
        self.encoder_eps = nn.Linear(hidden_units[-1], 1)

        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                pass
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        torch.nn.init.uniform_(self.linear.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        torch.nn.init.zeros_(self.linear.bias)

        # torch.nn.init.uniform_(self.classifier.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        # torch.nn.init.zeros_(self.classifier.bias)

        self.trunk.train()
        self.linear.train()
        self.classifier.train()
        self.encoder_eps.train()
        self.L1Loss = nn.L1Loss(reduction='none').to(device)
        self.CE_loss = nn.CrossEntropyLoss(reduction='none').to(device)

    def forward(self, x):
        x = self.trunk(x)
        d = self.linear(x)
        eps = self.encoder_eps(x)
        c = torch.softmax(self.classifier(x), -1)
        return d, eps, torch.clamp(c, 1e-20, torch.inf)

    def predict_disc_reward(self, reward_t, obs, obs_disc):
        label_eps = obs[:, -self.dim_c - 1].clone().unsqueeze(-1)
        label_c = obs[:, -self.dim_c:].clone()
        label_c = F.one_hot(torch.argmax(label_c, dim=-1), num_classes=self.dim_c)

        obs_disc = obs_disc[:, -self.disc_obs_len:, :].view(len(obs_disc), -1)
        with (torch.no_grad()):
            self.eval()
            if self.normalizer is not None:
                obs_disc = self.normalizer.normalize_torch(obs_disc, self.device)

            d, eps, c = self.forward(obs_disc)

            if self.disc_loss_function == "BCEWithLogitsLoss":
                reward_i = -torch.log(
                    torch.maximum(1 - 1 / (1 + torch.exp(-d)), torch.tensor(0.0001, device=self.device)))
            elif self.disc_loss_function == "MSELoss":
                reward_i = torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            elif self.disc_loss_function == "WassersteinLoss":
                reward_i = self.reward_i_normalizer.normalize_torch(d, self.device)
                self.reward_i_normalizer.update(d.cpu().numpy())
            else:
                raise ValueError("Unexpected style reward mapping specified")

            # unsupervised reward
            reward_us = -self.L1Loss(eps, label_eps)
            # semi-supervised reward
            reward_ss = -self.CE_loss(c, label_c.to(float)).unsqueeze(1)

            reward_i *= self.dt
            reward_us *= self.dt
            reward_ss *= self.dt

            rewards = (self.reward_i_coef * reward_i + self.reward_us_coef * reward_us +
                       self.reward_ss_coef * reward_ss + self.reward_t_coef * reward_t)

        return rewards.squeeze(), reward_i.squeeze(), reward_us.squeeze(), reward_ss.squeeze(), reward_t.squeeze()
