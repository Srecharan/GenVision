import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root='data', 
        train=True, 
        transform=transform,
        download=True
    )
    
    val_dataset = datasets.CIFAR10(
        root='data', 
        train=False, 
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

def preprocess_data(x):
    """Convert input data to range [-1, 1]"""
    return (2 * x - 1).to('cuda')