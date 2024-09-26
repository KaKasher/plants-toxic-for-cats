from torchvision.datasets import ImageFolder
import json

temp_dataset = ImageFolder('../data/train_test/train')

class_to_idx = temp_dataset.class_to_idx

idx_to_class = {v: k for k, v in class_to_idx.items()}

with open('idx_to_class.json', 'w') as f:
    json.dump(idx_to_class, f)
