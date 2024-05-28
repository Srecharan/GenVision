import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from cleanfid import fid

def save_samples(samples, filename, nrow=8):
    """Save generated samples as a grid image."""
    samples = (samples + 1) / 2  # Denormalize to [0, 1]
    grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
    torchvision.utils.save_image(grid, filename)

def interpolate_latent_space(generator, filename, n_samples=10):
    """Generate and save latent space interpolation."""
    device = next(generator.parameters()).device
    
    # Create interpolation points
    z1 = torch.randn(1, generator.latent_dim, device=device)
    z2 = torch.randn(1, generator.latent_dim, device=device)
    alpha = torch.linspace(0, 1, n_samples).view(-1, 1).to(device)
    
    interpolated_z = z1 * (1 - alpha) + z2 * alpha
    
    with torch.no_grad():
        samples = generator.forward_given_samples(interpolated_z)
        samples = (samples + 1) / 2  # Denormalize to [0, 1]
        save_samples(samples, filename, nrow=n_samples)

def get_fid(generator, dataset_name, dataset_resolution, batch_size=50, num_gen=50000):
    """Calculate FID score."""
    def gen_fn(z):
        with torch.no_grad():
            return generator.forward_given_samples(z)
    
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        batch_size=batch_size,
        verbose=True
    )
    return score

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count