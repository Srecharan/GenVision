import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Tuple, Optional

class CUBDataset(Dataset):
    """CUB-200-2011 Birds Dataset"""
    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load metadata
        self._load_metadata()
        
    def _load_metadata(self):
        """Load CUB-200-2011 metadata files"""
        # Load image paths and labels
        images_path = os.path.join(self.root_dir, 'images.txt')
        labels_path = os.path.join(self.root_dir, 'image_class_labels.txt')
        split_path = os.path.join(self.root_dir, 'train_test_split.txt')
        
        # Read files
        with open(images_path, 'r') as f:
            images = [line.strip().split() for line in f.readlines()]
        
        with open(labels_path, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]
            
        with open(split_path, 'r') as f:
            splits = [line.strip().split() for line in f.readlines()]
        
        # Create mappings
        self.image_paths = {}
        self.image_labels = {}
        self.train_test_split = {}
        
        for img_id, img_path in images:
            self.image_paths[img_id] = img_path
            
        for img_id, label in labels:
            self.image_labels[img_id] = int(label) - 1  # Convert to 0-indexed
            
        for img_id, is_train in splits:
            self.train_test_split[img_id] = int(is_train) == 1
        
        # Filter by split
        self.filtered_ids = []
        for img_id in self.image_paths.keys():
            if self.split == 'train' and self.train_test_split[img_id]:
                self.filtered_ids.append(img_id)
            elif self.split == 'test' and not self.train_test_split[img_id]:
                self.filtered_ids.append(img_id)
        
        print(f"Loaded {len(self.filtered_ids)} images for {self.split} split")
        
    def __len__(self):
        return len(self.filtered_ids)
    
    def __getitem__(self, idx):
        img_id = self.filtered_ids[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, 'images', self.image_paths[img_id])
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        label = self.image_labels[img_id]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_cub_dataloaders(root_dir: str, batch_size: int = 32, image_size: int = 224, num_workers: int = 4):
    """Create CUB-200-2011 data loaders for training and testing"""
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CUBDataset(root_dir, split='train', transform=train_transform)
    test_dataset = CUBDataset(root_dir, split='test', transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def download_cub_dataset(root_dir: str):
    """Download CUB-200-2011 dataset if not present"""
    import tarfile
    import urllib.request
    from tqdm import tqdm
    
    if os.path.exists(os.path.join(root_dir, 'CUB_200_2011')):
        print("CUB-200-2011 dataset already exists")
        return
    
    print("Downloading CUB-200-2011 dataset...")
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    
    os.makedirs(root_dir, exist_ok=True)
    tar_path = os.path.join(root_dir, 'CUB_200_2011.tgz')
    
    # Download with progress bar
    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(url, tar_path, reporthook=t.update_to)
    
    print("Extracting dataset...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        def extract_progress(members):
            for member in tqdm(members, desc="Extracting"):
                yield member
        tar.extractall(root_dir, members=extract_progress(tar))
    
    os.remove(tar_path)
    print("CUB-200-2011 dataset ready!")

if __name__ == "__main__":
    # Test the dataset loader
    root_dir = "data/CUB_200_2011"
    
    # Download dataset if needed
    download_cub_dataset("data")
    
    # Test data loaders
    train_loader, test_loader = get_cub_dataloaders(root_dir)
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Test a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label range: {labels.min()}-{labels.max()}")
        break 