import torch
from tqdm import tqdm
from .base import BaseDiffusion, extract
import numpy as np

class DDIM(BaseDiffusion):
    def __init__(self, *args, ddim_sampling_eta=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ddim_sampling_eta = ddim_sampling_eta

    def get_sampling_timesteps(self):
        """Get evenly spaced timesteps for DDIM sampling."""
        return np.linspace(
            0, self.num_timesteps - 1, 
            self.sampling_timesteps + 1
        ).round()[::-1][:-1]

    @torch.no_grad()
    def p_sample_ddim(self, x_t, t, t_prev):
        """Single step denoising function for DDIM."""
        # Predict noise and x_0
        pred_noise, x_0 = self.predict_noise(x_t, t)
        
        # Extract alpha values
        alpha_cumprod_t = extract(self.alphas_cumprod, t, x_t.shape)
        alpha_cumprod_t_prev = extract(self.alphas_cumprod, t_prev, x_t.shape)

        # Compute variance
        variance = self.ddim_sampling_eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
            (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )

        # Compute mean
        mean = torch.sqrt(alpha_cumprod_t_prev) * x_0 + \
               torch.sqrt(1 - alpha_cumprod_t_prev - variance**2) * pred_noise

        # Add noise scaled by variance
        nonzero_mask = (t != 0).float().reshape(-1, *([1] * (len(x_t.shape) - 1)))
        noise = torch.randn_like(x_t)
        return mean + nonzero_mask * variance * noise

    @torch.no_grad()
    def sample(self, shape):
        """Sample new images using DDIM."""
        device = next(self.model.parameters()).device
        b = shape[0]
        
        # Get timesteps for sampling
        timesteps = self.get_sampling_timesteps()
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        # Gradually denoise
        for i in tqdm(range(len(timesteps)), desc='DDIM Sampling'):
            t = torch.full((b,), timesteps[i], device=device, dtype=torch.long)
            t_prev = torch.full(
                (b,), 
                timesteps[i+1] if i < len(timesteps)-1 else 0, 
                device=device, 
                dtype=torch.long
            )
            img = self.p_sample_ddim(img, t, t_prev)
            
        return (img + 1) * 0.5  # Rescale to [0, 1]

    def forward(self, x_0, noise=None):
        """Forward diffusion process (same as DDPM)."""
        b, *_ = x_0.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x_0.device)
        noise = noise if noise is not None else torch.randn_like(x_0)

        noisy = (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
        
        return self.model(noisy, t), noise