import torch
import torch.nn as nn
from torch.nn import init

class DownSampleConv2D(nn.Module):
    def __init__(self, input_channels, kernel_size=3, n_filters=128, 
                 downscale_ratio=2, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, 
                             padding=padding)
        self.downscale_ratio = downscale_ratio
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_ratio)

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        chunks = torch.chunk(x, x.shape[1] // (self.downscale_ratio**2), dim=1)
        x = torch.stack(chunks, dim=2)
        x = x.permute(1, 0, 2, 3, 4)
        x = torch.mean(x, dim=0)
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, 
                     stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, 
                     stride=1, padding=1)
        )

    def forward(self, x):
        return self.layers(x) + x

class ResBlockDown(nn.Module):
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, 
                     stride=1, padding=1),
            nn.ReLU(),
            DownSampleConv2D(n_filters, kernel_size=3, n_filters=n_filters, padding=1)
        )
        self.downsample_residual = DownSampleConv2D(input_channels, kernel_size=1, 
                                                   n_filters=n_filters)

    def forward(self, x):
        return self.layers(x) + self.downsample_residual(x)

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.dense = nn.Linear(128, 1)
        self.layers = nn.Sequential(
            ResBlockDown(input_channels=channels),
            ResBlockDown(input_channels=128),
            ResBlock(input_channels=128),
            ResBlock(input_channels=128),
            nn.ReLU()
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

    def forward(self, x):
        x = self.layers(x)
        x = torch.sum(x, dim=(2, 3))
        return self.dense(x)