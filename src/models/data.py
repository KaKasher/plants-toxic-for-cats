import os
from pathlib import Path
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torch

def get_species_names(train_path):
    new_species_names = [name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))]
    return sorted(new_species_names)

def setup_data(batch_size=32):
    data_path = Path("../../data/train_test")
    train_path = data_path / "train"
    test_path = data_path / "test"

    transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.class_to_idx
