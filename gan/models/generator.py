import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class UpSampleConv2D(nn.Module):
    def __init__(self, input_channels, kernel_size=3, n_filters=128, 
                 upscale_factor=2, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, 
                             padding=padding)
        self.upscale_factor = upscale_factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, x):
        x = torch.repeat_interleave(x, self.upscale_factor**2, dim=1)
        x = self.pixel_shuffle(x)
        return self.conv(x)

class ResBlockUp(nn.Module):
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super().__init__()
        self.upsample_residual = UpSampleConv2D(input_channels, kernel_size=1, 
                                               n_filters=n_filters)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            UpSampleConv2D(n_filters, kernel_size=3, n_filters=n_filters, padding=1)
        )

    def forward(self, x):
        return self.layers(x) + self.upsample_residual(x)

class Generator(nn.Module):
    def __init__(self, starting_image_size=4, latent_dim=128, channels=3):
        super().__init__()
        self.starting_image_size = starting_image_size
        self.latent_dim = latent_dim
        self.channels = channels

        self.dense = nn.Linear(self.latent_dim, 2048)
        self.layers = nn.Sequential(
            ResBlockUp(128),
            ResBlockUp(128),
            ResBlockUp(128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.BatchNorm2d):
            init.normal_(module.weight.data, 1.0, 0.02)
            init.constant_(module.bias.data, 0)

    def forward_given_samples(self, z):
        x = self.dense(z)
        x = x.reshape(-1, 128, self.starting_image_size, self.starting_image_size)
        return self.layers(x)

    def forward(self, n_samples: int = 1024):
        z = torch.randn(n_samples, self.latent_dim, device=self.dense.weight.device)
        return self.forward_given_samples(z)