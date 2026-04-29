import numpy as np

import code
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
import torch.nn.functional as F


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class Actor(nn.Module):
    def __init__(self, num_prop,
                 num_auxiliary,
                 num_scan, 
                 num_actions_d,
                 num_actions_c,
                 scan_encoder_dims,
                 actor_hidden_dims, 
                 priv_encoder_dims, 
                 num_priv_latent, 
                 num_priv_explicit, 
                 num_hist, activation, 
                 tanh_encoder_output=False) -> None:
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_prop = num_prop
        self.num_auxiliary = num_auxiliary
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions_d = num_actions_d
        self.num_actions_c = num_actions_c
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
            priv_encoder_layers.append(activation)
            for l in range(len(priv_encoder_dims)):
                if l == len(priv_encoder_dims) - 1:
                    priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], num_priv_latent))
                    priv_encoder_layers.append(activation)
                else:
                    priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                    priv_encoder_layers.append(activation)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
        else:
            self.priv_encoder = nn.Identity()

        self.history_encoder = StateHistoryEncoder(activation, num_prop - num_auxiliary, num_hist, num_priv_latent)

        if self.if_scan_encode:
            scan_encoder = []
            scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
            scan_encoder.append(activation)
            for l in range(len(scan_encoder_dims) - 1):
                if l == len(scan_encoder_dims) - 2:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(activation)
            self.scan_encoder = nn.Sequential(*scan_encoder)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan
        
        actor_layers = []
        actor_layers.append(nn.Linear(num_prop+
                                      self.scan_encoder_output_dim+
                                      num_priv_explicit+
                                      num_priv_latent,
                                      actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
            actor_layers.append(activation)

        self.actor_trunk = nn.Sequential(*actor_layers)
        self.actor_d = nn.Linear(actor_hidden_dims[-1], num_actions_d)
        self.actor_c = nn.Linear(actor_hidden_dims[-1], num_actions_d * num_actions_c)

        # if tanh_encoder_output:
        #     actor_layers.append(nn.Tanh())
        # self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(self, obs, hist_encoding: bool, eval=False, scandots_latent=None):
        if self.if_scan_encode:
            obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
            if scandots_latent is None:
                scan_latent = self.scan_encoder(obs_scan)
            else:
                scan_latent = scandots_latent
            obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
        else:
            obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
        obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
        if hist_encoding:
            latent = self.infer_hist_latent(obs)
        else:
            latent = self.infer_priv_latent(obs)
        trunk_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
        trunk_output = self.actor_trunk(trunk_input)
        return trunk_output
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist * (self.num_prop - self.num_auxiliary):]
        return self.history_encoder(hist.view(-1, self.num_hist, (self.num_prop - self.num_auxiliary)))
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)

class ActorCriticTSC(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_auxiliary,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions_d,
                        num_actions_c,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        fixed_std=False,
                        device=torch.device('cuda'),
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticTSC, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        activation = get_activation(activation)
        self.num_actions_d = num_actions_d
        
        self.actor = Actor(num_prop, num_auxiliary, num_scan, num_actions_d, num_actions_c, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=kwargs['tanh_encoder_output'])
        

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        std = init_noise_std * torch.ones(num_actions_d * num_actions_c)
        self.std = torch.tensor(std).to(device) if fixed_std else nn.Parameter(std)
        self.distribution_d = None
        self.distribution_c = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
    
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
        return self.distribution_c.mean

    @property
    def action_std(self):
        return self.distribution_c.stddev
    
    @property
    def entropy_c(self):
        return self.distribution_c.entropy().mean(dim=-1)

    @property
    def entropy_d(self):
        return self.distribution_d.entropy()

    def act(self, observations, hist_encoding=False, **kwargs):
        embedding = self.actor(observations, hist_encoding)
        prob = torch.softmax(self.actor.actor_d(embedding), dim=-1)
        self.distribution_d = Categorical(probs=prob)
        actions_d = self.distribution_d.sample()
        mean = self.actor.actor_c(embedding)
        self.distribution_c = Normal(mean, mean*0. + self.std)
        actions_c = self.distribution_c.sample()
        actions = torch.cat([actions_d.unsqueeze(-1), actions_c], dim=-1)
        return actions
    
    def get_actions_log_prob_d(self, actions):
        return self.distribution_d.log_prob(actions)

    def get_actions_log_prob_c(self, actions):
        return self.distribution_c.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, **kwargs):
        embedding = self.actor(observations, hist_encoding, eval, scandots_latent)
        prob = torch.softmax(self.actor.actor_d(embedding), dim=-1)
        actions_d = torch.argmax(prob, dim=-1)
        actions_c = self.actor.actor_c(embedding)
        actions = torch.cat([actions_d.unsqueeze(-1), actions_c], dim=-1)
        return actions

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data


class ActorCriticBBC(nn.Module):
    is_recurrent = False

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 num_prop, num_auxiliary, num_hist, num_explicit, num_latent, num_command,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 priv_encoder_dims=[256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 fixed_std=False,
                 train_with_estimated_latent=True,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCriticBBC, self).__init__()

        activation = get_activation(activation)

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        self.train_with_estimated_latent = train_with_estimated_latent
        self.num_prop = num_prop - num_auxiliary
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

        self.history_encoder = StateHistoryEncoder(activation, self.num_prop, num_hist, num_latent)

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

        # Action noise
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
