from torchvision.models import resnet34
from torch import nn
from utils import load_model
from pathlib import Path


def setup_model(num_new_classes):
    plantnet_model_path = Path("../../models/plantnet_resnet34_best.tar")
    plantnet_model = resnet34(num_classes=1081)
    load_model(plantnet_model, filename=plantnet_model_path, use_gpu=True)

    # freeze all the layers
    for param in plantnet_model.parameters():
        param.requires_grad = False

    # replace the final layer
    num_features = plantnet_model.fc.in_features
    plantnet_model.fc = nn.Linear(num_features, num_new_classes)

    return plantnet_model