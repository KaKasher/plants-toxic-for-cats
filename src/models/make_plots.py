import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import timm
import argparse
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Define paths
MODELS_DIR = Path("../../models")
DATA_DIR = Path("../../data/train_test")

def load_losses(model_name):
    # Locate the json file
    files = list((MODELS_DIR / "losses_metrics").glob(f"{model_name}*losses.json"))
    if not files:
        print(f"No loss file found for {model_name}")
        return None
    with open(files[0], 'r') as f:
        return json.load(f)

def plot_comparison(resnet_losses, vit_losses):
    plt.figure(figsize=(12, 6))
    
    # Plot ResNet
    epochs_r = range(1, len(resnet_losses['train_losses']) + 1)
    plt.plot(epochs_r, resnet_losses['train_losses'], 'b--', alpha=0.6, label='ResNet Train')
    plt.plot(epochs_r, resnet_losses['test_losses'], 'b-', label='ResNet Val')

    # Plot ViT
    epochs_v = range(1, len(vit_losses['train_losses']) + 1)
    plt.plot(epochs_v, vit_losses['train_losses'], 'r--', alpha=0.6, label='ViT Train')
    plt.plot(epochs_v, vit_losses['test_losses'], 'r-', label='ViT Val')

    plt.title('Training Dynamics: ResNet152 vs ViT-Base')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Weighted Cross Entropy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = MODELS_DIR / "plots" / "model_comparison.png"
    plt.savefig(out_path)
    print(f"Comparison plot saved to {out_path}")
    plt.close()

def plot_confusion_matrix(model_path, classes):
    # Load model (assuming ViT for the confusion matrix as it is the best model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Re-create model structure (ViT)
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(classes))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Setup data loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder(DATA_DIR / "test", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []

    print("Generating predictions for Confusion Matrix...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (ViT-Base)')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    # Add counts
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    out_path = MODELS_DIR / "plots" / "confusion_matrix.png"
    plt.savefig(out_path)
    print(f"Confusion matrix saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    # 1. Comparison Plot
    r_losses = load_losses("resnet152")
    v_losses = load_losses("vit_b16")
    
    if r_losses and v_losses:
        plot_comparison(r_losses, v_losses)

    # 2. Confusion Matrix (ViT only)
    # Find ViT model file
    vit_files = list(MODELS_DIR.glob("vit_b16*.tar"))
    if vit_files:
        # Get class names
        class_names = sorted([d.name for d in (DATA_DIR / "train").iterdir() if d.is_dir()])
        plot_confusion_matrix(vit_files[0], class_names)
    else:
        print("No ViT model file found for confusion matrix.")
