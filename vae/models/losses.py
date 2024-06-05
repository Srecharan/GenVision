import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss:
    def __init__(self, beta=1.0):
        self.beta = beta
        
    def __call__(self, model, x):
        recon_x, mu, log_std = model(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='none')
        recon_loss = recon_loss.sum(dim=(1, 2, 3)).mean()
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std), dim=1)
        kl_loss = kl_loss.mean()
        
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, {'recon_loss': recon_loss, 'kl_loss': kl_loss}

class AutoEncoderLoss:
    def __call__(self, model, x):
        recon_x = model(x)
        loss = F.mse_loss(recon_x, x, reduction='none')
        loss = loss.sum(dim=(1, 2, 3)).mean()
        return loss, {'recon_loss': loss}

class BetaScheduler:
    def __init__(self, mode='constant', max_epochs=None, target_val=1.0):
        self.mode = mode
        self.max_epochs = max_epochs
        self.target_val = target_val
        
    def __call__(self, epoch):
        if self.mode == 'constant':
            return self.target_val
        elif self.mode == 'linear':
            return self.target_val * (epoch / self.max_epochs)
        else:
            raise ValueError(f"Unknown beta scheduling mode: {self.mode}")