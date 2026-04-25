import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with 'real' and 'fake' subfolders.
            transform: Albumentations transformations.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load real images (label 0)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for img in os.listdir(real_dir):
                self.image_paths.append(os.path.join(real_dir, img))
                self.labels.append(0)

        # Load fake images (label 1)
        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            for img in os.listdir(fake_dir):
                self.image_paths.append(os.path.join(fake_dir, img))
                self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        import numpy as np
        image = np.array(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

def get_train_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
