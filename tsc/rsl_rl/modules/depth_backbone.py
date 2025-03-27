import torch
import torch.nn as nn
import sys
import torchvision
from rsl_rl.modules.byol import BYOL

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, n_depth_latent, env_cfg) -> None:
        super().__init__()
        self.n_delta_yaw = env_cfg.env.n_delta_yaw
        self.n_obst_type = env_cfg.env.n_obst_type
        self.n_depth_latent = n_depth_latent
        activation = nn.ELU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.base_backbone = base_backbone

        self.byol_learner = BYOL(
            self.base_backbone,
            image_size=(58, 87),
            hidden_layer=-1
        )

        self.combination_mlp = nn.Sequential(
                                    nn.Linear(self.n_depth_latent + env_cfg.env.n_proprio, 128),
                                    activation,
                                    nn.Linear(128, self.n_depth_latent)
                                )
        self.rnn = nn.GRU(input_size=self.n_depth_latent, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, self.n_depth_latent+self.n_delta_yaw+self.n_obst_type)  # latent, delta_yaw, obst_type
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        depth_latent = torch.cat([depth_latent[:, :self.n_depth_latent+self.n_delta_yaw],
                                  self.softmax(depth_latent[:, self.n_depth_latent+self.n_delta_yaw:])], dim=-1)
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

class StackDepthEncoder(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )

        self.conv1d = nn.Sequential(nn.Conv1d(in_channels=env_cfg.depth.buffer_len, out_channels=16, kernel_size=4, stride=2),  # (30 - 4) / 2 + 1 = 14,
                                    activation,
                                    nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2), # 14-2+1 = 13,
                                    activation)
        self.mlp = nn.Sequential(nn.Linear(16*14, 32), 
                                 activation)
        
    def forward(self, depth_image, proprioception):
        # depth_image shape: [batch_size, num, 58, 87]
        depth_latent = self.base_backbone(None, depth_image.flatten(0, 1), None)  # [batch_size * num, 32]
        depth_latent = depth_latent.reshape(depth_image.shape[0], depth_image.shape[1], -1)  # [batch_size, num, 32]
        depth_latent = self.conv1d(depth_latent)
        depth_latent = self.mlp(depth_latent.flatten(1, 2))
        return depth_latent

    
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

        self.augment = None

    def forward(self, images: torch.Tensor):
        if self.augment:
            images = self.augment(images.clone())
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent