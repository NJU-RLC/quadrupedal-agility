import torch
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, dim_c, history_len, buffer_size, device):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.states = torch.zeros(buffer_size, history_len * obs_dim).to(device)
        self.latent_eps = torch.zeros(buffer_size, 1).to(device)
        self.latent_c = torch.zeros(buffer_size, dim_c).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states, latent_eps, latent_c):
        """Add new states to memory."""

        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.states[self.step:self.buffer_size] = states[:self.buffer_size - self.step].clone()
            self.latent_eps[self.step:self.buffer_size] = latent_eps[:self.buffer_size - self.step].clone()
            self.latent_c[self.step:self.buffer_size] = latent_c[:self.buffer_size - self.step].clone()
            self.states[:end_idx - self.buffer_size] = states[self.buffer_size - self.step:].clone()
            self.latent_eps[:end_idx - self.buffer_size] = latent_eps[self.buffer_size - self.step:].clone()
            self.latent_c[:end_idx - self.buffer_size] = latent_c[self.buffer_size - self.step:].clone()
        else:
            self.states[start_idx:end_idx] = states.clone()
            self.latent_eps[start_idx:end_idx] = latent_eps.clone()
            self.latent_c[start_idx:end_idx] = latent_c.clone()

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (self.states[sample_idxs].to(self.device),
                   self.latent_eps[sample_idxs].to(self.device),
                   self.latent_c[sample_idxs].to(self.device))
