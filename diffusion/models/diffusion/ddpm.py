import torch
from tqdm import tqdm
from .base import BaseDiffusion, extract

class DDPM(BaseDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """Single step denoising function for DDPM."""
        mean, var = self.p_mean_variance(x_t, t)
        noise = torch.randn_like(x_t)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float().reshape(-1, *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, shape):
        """Sample new images using DDPM."""
        device = next(self.model.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        # Gradually denoise
        for timestep in tqdm(
            reversed(range(self.num_timesteps)), 
            desc='DDPM Sampling',
            total=self.num_timesteps
        ):
            t = torch.full((b,), timestep, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
            
        return (img + 1) * 0.5  # Rescale to [0, 1]

    def forward(self, x_0, noise=None):
        """Forward diffusion process."""
        b, *_ = x_0.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x_0.device)
        noise = noise if noise is not None else torch.randn_like(x_0)

        # Get noisy image
        noisy = (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
        
        return self.model(noisy, t), noise