# models/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        channels, height, width = input_shape
        self.convs = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        
        self.flatten_size = 256 * (height // 8) * (width // 8)
        self.fc = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

class VAEEncoder(BaseEncoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        self.fc = nn.Linear(self.flatten_size, 2 * latent_dim)
        
    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        mean, log_std = torch.chunk(x, 2, dim=1)
        return mean, log_std
