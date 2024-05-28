import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from glob import glob

class Dataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.image_files = []
        
        # Collect all image files
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.image_files.extend(glob(os.path.join(root, '**', ext), recursive=True))
        
        if len(self.image_files) == 0:
            raise RuntimeError(f'Found 0 images in {root}')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        img = default_loader(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img