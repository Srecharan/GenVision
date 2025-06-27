import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json

from models.resnet_classifier import create_classifier
from utils.cub_dataset import get_cub_dataloaders
from utils.metrics import AverageMeter, accuracy, save_results

class ClassificationTrainer:
    """Trainer for bird species classification with synthetic data augmentation"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = create_classifier(
            model_type=args.model_type,
            num_classes=200,  # CUB-200-2011 has 200 bird species
            pretrained=args.pretrained
        ).to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.best_acc = 0.0
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Setup logging
        self.writer = SummaryWriter(args.log_dir)
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            # Update metrics
            losses.update(loss.item(), data.size(0))
            top1_acc.update(acc1[0], data.size(0))
            top5_acc.update(acc5[0], data.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Top1': f'{top1_acc.avg:.2f}%',
                'Top5': f'{top5_acc.avg:.2f}%'
            })
        
        # Log to tensorboard
        self.writer.add_scalar('Train/Loss', losses.avg, epoch)
        self.writer.add_scalar('Train/Top1_Accuracy', top1_acc.avg, epoch)
        self.writer.add_scalar('Train/Top5_Accuracy', top5_acc.avg, epoch)
        
        return losses.avg, top1_acc.avg, top5_acc.avg
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        
        losses = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Calculate accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
                # Update metrics
                losses.update(loss.item(), data.size(0))
                top1_acc.update(acc1[0], data.size(0))
                top5_acc.update(acc5[0], data.size(0))
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', losses.avg, epoch)
        self.writer.add_scalar('Val/Top1_Accuracy', top1_acc.avg, epoch)
        self.writer.add_scalar('Val/Top5_Accuracy', top5_acc.avg, epoch)
        
        return losses.avg, top1_acc.avg, top5_acc.avg
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'val_acc': val_acc,
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.args.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(state, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(state, best_path)
            print(f"New best model saved with accuracy: {val_acc:.2f}%")
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"Starting training for {self.args.epochs} epochs")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            # Train
            train_loss, train_top1, train_top5 = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_top1, val_top5 = self.validate(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_top1)
            self.val_accs.append(val_top1)
            
            # Check if best model
            is_best = val_top1 > self.best_acc
            if is_best:
                self.best_acc = val_top1
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_top1, is_best)
            
            # Print epoch summary
            print(f"Epoch [{epoch+1}/{self.args.epochs}]")
            print(f"Train - Loss: {train_loss:.4f}, Top1: {train_top1:.2f}%, Top5: {train_top5:.2f}%")
            print(f"Val   - Loss: {val_loss:.4f}, Top1: {val_top1:.2f}%, Top5: {val_top5:.2f}%")
            print(f"Best Val Accuracy: {self.best_acc:.2f}%")
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_acc:.2f}%")
        
        # Save final results
        results = {
            'best_val_accuracy': self.best_acc,
            'final_train_accuracy': train_top1,
            'training_time_seconds': total_time,
            'epochs_completed': self.args.epochs,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accs,
            'val_accuracies': self.val_accs
        }
        
        results_path = os.path.join(self.args.results_dir, 'training_results.json')
        save_results(results, results_path)
        
        self.writer.close()
        return results

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet-50 classifier for CUB-200-2011')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='resnet50', 
                        choices=['resnet50', 'efficientnet'],
                        help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='data/CUB_200_2011',
                        help='Path to CUB-200-2011 dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--step_size', type=int, default=15,
                        help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Scheduler gamma')
    
    # Synthetic data arguments
    parser.add_argument('--synthetic_data_dir', type=str, default=None,
                        help='Path to synthetic data directory')
    parser.add_argument('--synthetic_ratio', type=float, default=1.0,
                        help='Ratio of synthetic to real data')
    parser.add_argument('--data_fraction', type=float, default=1.0,
                        help='Fraction of training data to use')
    
    # Output arguments
    parser.add_argument('--log_dir', type=str, default='logs/classification',
                        help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/classification',
                        help='Directory for model checkpoints')
    parser.add_argument('--results_dir', type=str, default='results/classification',
                        help='Directory for results')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--experiment_name', type=str, default='baseline',
                        help='Experiment name for logging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("Loading CUB-200-2011 dataset...")
    train_loader, test_loader = get_cub_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create trainer
    trainer = ClassificationTrainer(args)
    
    # Train model
    results = trainer.train(train_loader, test_loader)
    
    print(f"Training completed! Best accuracy: {results['best_val_accuracy']:.2f}%")

if __name__ == '__main__':
    main() 