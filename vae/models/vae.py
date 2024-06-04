import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import BaseEncoder, VAEEncoder
from .decoder import Decoder

class BaseVAE(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), latent_dim=128):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
    def encode(self, x):
        raise NotImplementedError
        
    def decode(self, z):
        raise NotImplementedError
        
    def reparameterize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        raise NotImplementedError

class AutoEncoder(BaseVAE):
    def __init__(self, input_shape=(3, 32, 32), latent_dim=128):
        super().__init__(input_shape, latent_dim)
        self.encoder = BaseEncoder(input_shape, latent_dim)
        self.decoder = Decoder(latent_dim, input_shape)
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class VariationalAutoEncoder(BaseVAE):
    def __init__(self, input_shape=(3, 32, 32), latent_dim=128):
        super().__init__(input_shape, latent_dim)
        self.encoder = VAEEncoder(input_shape, latent_dim)
        self.decoder = Decoder(latent_dim, input_shape)
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, log_std = self.encode(x)
        z = self.reparameterize(mu, log_std)
        return self.decode(z), mu, log_std
        
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
        return self.decode(z)