# models/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        
        channels, height, width = output_shape
        self.base_size = 256 * (height // 8) * (width // 8)
        
        self.fc = nn.Linear(latent_dim, self.base_size)
        
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.reshape(-1, 256, self.output_shape[1] // 8, self.output_shape[2] // 8)
        return self.deconvs(x)