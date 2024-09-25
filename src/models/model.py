from torchvision.models import resnet152
from torch import nn
from utils import load_model, get_model
from pathlib import Path
import timm


def setup_model(num_new_classes, model_name="resnet152"):
    models_folder = Path("../../models")

    if model_name == "resnet152":
        plantnet_model_path = models_folder / "resnet152_weights_best_acc.tar"
        plantnet_model = resnet152(num_classes=1081)
        load_model(plantnet_model, filename=plantnet_model_path, use_gpu=True)

        for param in plantnet_model.parameters():
            param.requires_grad = False

        num_features = plantnet_model.fc.in_features
        plantnet_model.fc = nn.Linear(num_features, num_new_classes)
    elif model_name == "efficientnet_b4":
        plantnet_model_path = models_folder / "efficientnet_b4_weights_best_acc.tar"
        plantnet_model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1081)
        load_model(plantnet_model, filename=plantnet_model_path, use_gpu=True)

        for param in plantnet_model.parameters():
            param.requires_grad = False

        num_features = plantnet_model.classifier.in_features
        plantnet_model.classifier = nn.Linear(num_features, num_new_classes)
    elif model_name == "vit_b16_224":
        plantnet_model_path = models_folder / "vit_base_patch16_224_weights_best_acc.tar"
        plantnet_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1081)
        load_model(plantnet_model, filename=plantnet_model_path, use_gpu=True)

        for param in plantnet_model.parameters():
             param.requires_grad = False

        num_features = plantnet_model.head.in_features
        plantnet_model.head = nn.Linear(num_features, num_new_classes)
    else:
        raise ValueError(f"Model name {model_name} not supported")


    return plantnet_model