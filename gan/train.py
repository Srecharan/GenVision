import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import GANLoss, WGANGPLoss, LSGANLoss
from utils.dataset import Dataset
from utils.utils import save_samples, get_fid, interpolate_latent_space

class GANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.generator = Generator(
            starting_image_size=config.starting_image_size,
            latent_dim=config.latent_dim,
            channels=config.channels
        ).to(self.device)
        
        self.discriminator = Discriminator(
            channels=config.channels
        ).to(self.device)
        
        self.setup_optimizers()
        self.setup_loss()
        self.writer = SummaryWriter(config.log_dir)
        self.iteration = 0

    def setup_optimizers(self):
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr,
            betas=(0.0, 0.9)
        )
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr,
            betas=(0.0, 0.9)
        )
        
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.d_optimizer,
            lr_lambda=lambda x: max(0.0, 1.0 - x / 500000)
        )
        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.g_optimizer,
            lr_lambda=lambda x: max(0.0, 1.0 - x / 100000)
        )

    def setup_loss(self):
        if self.config.loss_type == 'wgan_gp':
            self.loss_module = WGANGPLoss(lambda_gp=self.config.lambda_gp)
        elif self.config.loss_type == 'lsgan':
            self.loss_module = LSGANLoss()
        else:
            self.loss_module = GANLoss()

    def train_discriminator(self, real_batch):
        self.d_optimizer.zero_grad()
        
        fake_images = self.generator(real_batch.size(0))
        d_real = self.discriminator(real_batch)
        d_fake = self.discriminator(fake_images.detach())
        
        if self.config.loss_type == 'wgan_gp':
            epsilon = torch.rand(real_batch.size(0), 1, 1, 1).to(self.device)
            interpolated = epsilon * real_batch + (1 - epsilon) * fake_images
            d_interp = self.discriminator(interpolated)
            d_loss = self.loss_module.compute_discriminator_loss(
                d_real, d_fake, d_interp, interpolated
            )
        else:
            d_loss = self.loss_module.compute_discriminator_loss(d_real, d_fake)
        
        d_loss.backward()
        self.d_optimizer.step()
        return d_loss.item()

    def train_generator(self, batch_size):
        self.g_optimizer.zero_grad()
        
        fake_batch = self.generator(batch_size)
        d_fake = self.discriminator(fake_batch)
        g_loss = self.loss_module.compute_generator_loss(d_fake)
        
        g_loss.backward()
        self.g_optimizer.step()
        return g_loss.item()

    def train_step(self, real_batch):
        d_loss = self.train_discriminator(real_batch)
        g_loss = None
        
        if self.iteration % self.config.n_critic == 0:
            g_loss = self.train_generator(real_batch.size(0))
            
        self.d_scheduler.step()
        self.g_scheduler.step()
        
        return {'d_loss': d_loss, 'g_loss': g_loss}

    def train(self, train_loader):
        for epoch in range(self.config.num_epochs):
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
            for real_batch in pbar:
                real_batch = real_batch.to(self.device)
                metrics = self.train_step(real_batch)
                
                if self.iteration % self.config.log_interval == 0:
                    self.log_training(metrics)
                    
                if self.iteration % self.config.sample_interval == 0:
                    self.generate_samples()
                    
                if self.iteration % self.config.save_interval == 0:
                    self.save_checkpoint()
                
                self.iteration += 1
                pbar.set_postfix(metrics)

    def log_training(self, metrics):
        for key, value in metrics.items():
            if value is not None:
                self.writer.add_scalar(f'train/{key}', value, self.iteration)

    def generate_samples(self):
        self.generator.eval()
        with torch.no_grad():
            samples = self.generator(64)
            save_samples(
                samples, 
                os.path.join(self.config.sample_dir, f'samples_{self.iteration}.png')
            )
            interpolate_latent_space(
                self.generator,
                os.path.join(self.config.sample_dir, f'interpolation_{self.iteration}.png')
            )
        self.generator.train()

    def save_checkpoint(self):
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'iteration': self.iteration
        }
        torch.save(
            checkpoint,
            os.path.join(self.config.checkpoint_dir, f'checkpoint_{self.iteration}.pt')
        )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_type', type=str, default='gan', 
                       choices=['gan', 'wgan_gp', 'lsgan'])
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--sample_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--starting_image_size', type=int, default=4)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = Dataset(root='data', transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    trainer = GANTrainer(args)
    trainer.train(dataloader)