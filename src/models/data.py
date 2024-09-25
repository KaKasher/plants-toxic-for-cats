import os
from pathlib import Path
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TransformsWrapper:
    # Wraps an albumentations transform to be used with torchvision datasets
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']

def get_species_names(train_path):
    new_species_names = [name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))]
    return sorted(new_species_names)

def setup_data(batch_size=32):
    data_path = Path("/home/kaka/repo/plants-toxic-for-cats/data/train_test")
    train_path = data_path / "train"
    test_path = data_path / "test"

    train_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    train_dataset = ImageFolder(train_path, transform=TransformsWrapper(transforms=test_transform))
    test_dataset = ImageFolder(test_path, transform=TransformsWrapper(transforms=test_transform))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
