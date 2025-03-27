import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size), self.activation_fn,
        )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=8, stride=4),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=5, stride=1),
                self.activation_fn,
                nn.Conv1d(in_channels=channel_size, out_channels=channel_size, kernel_size=5, stride=1),
                self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=4, stride=2),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=2, stride=1),
                self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=6, stride=2),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=4, stride=2),
                self.activation_fn,
                nn.Flatten())
        else:
            raise (ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, output_size), self.activation_fn
        )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1]))  # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 num_prop, num_hist, num_explicit, num_latent, num_command,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 priv_encoder_dims=[256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 fixed_std=False,
                 train_with_estimated_latent=False,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        self.train_with_estimated_latent = train_with_estimated_latent
        self.num_prop = num_prop
        self.num_explicit = num_explicit
        self.num_latent = num_latent
        self.num_hist = num_hist
        self.num_command = num_command

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_latent, priv_encoder_dims[0]))
            priv_encoder_layers.append(activation)
            for l in range(len(priv_encoder_dims)):
                if l == len(priv_encoder_dims) - 1:
                    priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], num_latent))
                    priv_encoder_layers.append(activation)
                else:
                    priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                    priv_encoder_layers.append(activation)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
        else:
            self.priv_encoder = nn.Identity()

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, num_latent)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
            actor_layers.append(activation)
        self.actor_trunk = nn.Sequential(*actor_layers)
        self.actor_head = nn.Linear(actor_hidden_dims[-1], num_actions)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
            critic_layers.append(activation)
        self.critic_trunk = nn.Sequential(*critic_layers)
        self.critic_head = nn.Linear(critic_hidden_dims[-1], 1)

        for m in self.actor_trunk.modules():
            if isinstance(m, nn.Linear):
                pass
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        # print(f"Actor MLP: {self.actor}")
        # print(f"Critic MLP: {self.critic}")

        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding: bool):
        obs_prop = observations[:, :self.num_prop]
        obs_explicit = observations[:, self.num_prop:self.num_prop + self.num_explicit]
        obs_latent = observations[:, self.num_prop + self.num_explicit:self.num_prop + self.num_explicit + self.num_latent]
        obs_hist = observations[:,
                   self.num_prop + self.num_explicit + self.num_latent:self.num_prop + self.num_explicit + self.num_latent + self.num_hist * self.num_prop]
        obs_command = observations[:, self.num_prop + self.num_explicit + self.num_latent + self.num_hist * self.num_prop:]

        if self.train_with_estimated_latent:
            if hist_encoding:
                obs_latent = self.infer_hist_latent(obs_hist)
            else:
                obs_latent = self.infer_priv_latent(obs_latent)

        observations = torch.cat([obs_prop, obs_explicit, obs_latent, obs_command], dim=-1)
        embedding = self.actor_trunk(observations)
        mean = self.actor_head(embedding)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=True):
        obs_prop = observations[:, :self.num_prop]
        obs_explicit = observations[:, self.num_prop:self.num_prop + self.num_explicit]
        obs_latent = observations[:, self.num_prop + self.num_explicit:self.num_prop + self.num_explicit + self.num_latent]
        obs_hist = observations[:,
                   self.num_prop + self.num_explicit + self.num_latent:self.num_prop + self.num_explicit + self.num_latent + self.num_hist * self.num_prop]
        obs_command = observations[:, self.num_prop + self.num_explicit + self.num_latent + self.num_hist * self.num_prop:]
        if self.train_with_estimated_latent:
            if hist_encoding:
                obs_latent = self.infer_hist_latent(obs_hist)
            else:
                obs_latent = self.infer_priv_latent(obs_latent)

        observations = torch.cat([obs_prop, obs_explicit, obs_latent, obs_command], dim=-1)
        embedding = self.actor_trunk(observations)
        mean = self.actor_head(embedding)
        return mean

    def infer_priv_latent(self, obs):
        return self.priv_encoder(obs)

    def infer_hist_latent(self, obs):
        return self.history_encoder(obs.view(-1, self.num_hist, self.num_prop))

    def evaluate(self, critic_observations, **kwargs):
        embedding = self.critic_trunk(critic_observations)
        value = self.critic_head(embedding)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
