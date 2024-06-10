import torch
import torch.nn as nn
from tqdm import tqdm
from ..unet.unet import Unet
from utils.scheduler import cosine_beta_schedule

class BaseDiffusion(nn.Module):
    def __init__(
        self,
        model: Unet,
        timesteps: int = 1000,
        sampling_timesteps: int = None,
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.device = torch.cuda.current_device()
        
        # Setup noise schedule
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps or timesteps

        # Calculate betas and alphas
        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.ones(1, device=self.device), 
            self.alphas_cumprod[:-1]
        ])

        # Pre-compute diffusion parameters
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        # DDPM posterior parameters
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / 
            (1 - self.alphas_cumprod)
        )

    def predict_noise(self, x_t, t, clip_x0=True):
        """Predict the noise added to the image."""
        pred_noise = self.model(x_t, t)
        x_0 = self.predict_x0_from_noise(x_t, t, pred_noise)
        
        if clip_x0:
            x_0.clamp_(-1, 1)
            
        return pred_noise, x_0

    def predict_x0_from_noise(self, x_t, t, noise):
        """Predict the original image from a noisy image and predicted noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_0, x_t, t):
        """Compute the posterior mean and variance."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_var = extract(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_var

    def p_mean_variance(self, x_t, t, clip_denoised=True):
        """Compute mean and variance for the denoising step."""
        pred_noise, x_0 = self.predict_noise(x_t, t)
        
        if clip_denoised:
            x_0.clamp_(-1, 1)

        return self.q_posterior(x_0, x_t, t)

    @torch.no_grad()
    def sample(self, shape):
        """Abstract method for sampling - to be implemented by subclasses."""
        raise NotImplementedError

def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps and reshape to match image dimensions."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))