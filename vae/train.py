import argparse
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from models.vae import AutoEncoder, VariationalAutoEncoder
from utils.data import get_dataloaders, preprocess_data
from utils.metrics import AverageMeter, compute_average_metrics

def ae_loss(model, x):
    """Compute MSE reconstruction loss for autoencoder"""
    z = model.encoder(x)
    recon = model.decoder(z)
    squared_diff = (recon - x) ** 2
    loss = torch.sum(squared_diff, dim=(1, 2, 3))
    loss = torch.mean(loss)
    return loss, OrderedDict(recon_loss=loss)

def vae_loss(model, x, beta=1):
    """Compute VAE loss with reconstruction and KL divergence terms"""
    z_mean, z_log_std = model.encoder(x)
    z_std = torch.exp(z_log_std)
    epsilon = torch.randn_like(z_std).to(x.device)
    z = z_mean + z_std * epsilon
    
    recon = model.decoder(z)
    squared_diff = (recon - x) ** 2
    recon_loss = torch.sum(squared_diff, dim=(1, 2, 3))
    recon_loss = torch.mean(recon_loss)
    
    kl_loss = 0.5 * torch.sum(
        z_mean**2 + z_std**2 - (2 * z_log_std + 1), 
        dim=1
    )
    kl_loss = torch.mean(kl_loss)
    
    total_loss = recon_loss + beta * kl_loss
    return total_loss, OrderedDict(recon_loss=recon_loss, kl_loss=kl_loss)

class BetaScheduler:
    def __init__(self, mode='constant', max_epochs=None, target_val=1):
        self.mode = mode
        self.max_epochs = max_epochs
        self.target_val = target_val

    def __call__(self, epoch):
        if self.mode == 'constant':
            return self.target_val
        elif self.mode == 'linear':
            return self.target_val * epoch / self.max_epochs
        else:
            raise ValueError(f"Unknown beta mode: {self.mode}")

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup model
        if args.loss_mode == "vae":
            self.model = VariationalAutoEncoder(
                input_shape=(3, 32, 32),
                latent_dim=args.latent_size
            ).to(self.device)
        else:
            self.model = AutoEncoder(
                input_shape=(3, 32, 32),
                latent_dim=args.latent_size
            ).to(self.device)
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.beta_scheduler = BetaScheduler(
            mode=args.beta_mode,
            max_epochs=args.num_epochs,
            target_val=args.target_beta_val
        )
        
        # Create directories
        self.log_dir = f"data/{args.log_dir}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir)

    def train_epoch(self, loader, beta):
        self.model.train()
        metrics_list = []
        
        for x, _ in loader:
            x = preprocess_data(x)
            self.optimizer.zero_grad()
            
            if self.args.loss_mode == "ae":
                loss, metrics = ae_loss(self.model, x)
            else:
                loss, metrics = vae_loss(self.model, x, beta)
                
            metrics_list.append(metrics)
            loss.backward()
            
            if self.args.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.args.grad_clip
                )
                
            self.optimizer.step()
            
        return compute_average_metrics(metrics_list)

    def validate(self, loader):
        self.model.eval()
        metrics_list = []
        
        with torch.no_grad():
            for x, _ in loader:
                x = preprocess_data(x)
                
                if self.args.loss_mode == "ae":
                    _, metrics = ae_loss(self.model, x)
                else:
                    _, metrics = vae_loss(self.model, x)
                    
                metrics_list.append(metrics)
        
        return compute_average_metrics(metrics_list)

    def save_metrics_plot(self, metrics_dict, metric_name):
        values = metrics_dict[metric_name]
        plt.figure()
        plt.plot(range(len(values)), values)
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs Epochs")
        plt.savefig(f"{self.log_dir}/{metric_name}_vs_iterations.png")
        plt.close()

    def train(self, train_loader, val_loader):
        plot_metrics = {}
        
        for epoch in range(self.args.num_epochs):
            print(f"Epoch {epoch}")
            beta = self.beta_scheduler(epoch)
            
            # Training
            train_metrics = self.train_epoch(train_loader, beta)
            val_metrics = self.validate(val_loader)
            
            # Logging
            for k, v in val_metrics.items():
                if k not in plot_metrics:
                    plot_metrics[k] = []
                plot_metrics[k].append(v)
                
            if (epoch + 1) % self.args.eval_interval == 0:
                print(f"Train metrics: {train_metrics}")
                print(f"Val metrics: {val_metrics}")
                
                for k in plot_metrics.keys():
                    self.save_metrics_plot(plot_metrics, k)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="vae_test")
    parser.add_argument("--loss_mode", type=str, default="ae")
    parser.add_argument("--beta_mode", type=str, default="constant")
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--target_beta_val", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Get data
    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size)
    
    # Train
    trainer = Trainer(args)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()