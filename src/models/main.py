import torch
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from model import setup_model
from data import setup_data, get_species_names
from train import train_test_loop
from metrics import calculate_metrics
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using pretrained plantnet model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--model_name", type=str, default="resnet152", help="Choose from 'resnet152', 'efficientnet_b4', 'vit_b16_224'")
    return parser.parse_args()

def save_losses(train_losses, test_losses, filename):
    loss_data = {
        "train_losses": train_losses,
        "test_losses": test_losses
    }
    with open(filename, "w") as f:
        json.dump(loss_data, f)

def plot_losses(train_losses, test_losses, filename):
    num_epochs = len(train_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory


def main():
    args = parse_args()

    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = Path("../../data/train_test")
    train_path = data_path / "train"
    species_names = get_species_names(train_path)
    num_classes = len(species_names)

    # Get pretrained model
    plantnet_model = setup_model(num_classes, model_name=args.model_name)
    train_loader, test_loader = setup_data(batch_size=args.batch_size)

    criterion = torch.nn.CrossEntropyLoss()

    # Setup tenorboard
    models_folder = Path("../../models")
    model_name = f"{args.model_name}_{args.epochs}e_{args.batch_size}bs_{args.lr}lr_adamW_hflip"
    tensorboard_dir = models_folder / "tensorboard" / model_name
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Train the model
    train_losses, test_losses = train_test_loop(plantnet_model, train_loader, test_loader, criterion, device, writer,
                                                args.epochs, args.lr)

    writer.close()

    model_save_path = models_folder / f"{model_name}.tar"
    torch.save(plantnet_model.state_dict(), model_save_path)

    # Save the losses
    loss_save_path = models_folder / f"losses_metrics/{model_name}_losses.json"
    save_losses(train_losses, test_losses, loss_save_path)

    # Calculate and save metrics
    metrics = calculate_metrics(plantnet_model, test_loader, device)
    metrics_save_path = models_folder / f"losses_metrics/{model_name}_metrics.json"
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f)

    # Plot and save the graph
    plot_save_path = models_folder / f"plots/{model_name}_plot.png"
    plot_losses(train_losses, test_losses, plot_save_path)

    print(f"Model saved to {model_save_path}")
    print(f"Loss data saved to {loss_save_path}")
    print(f"Loss plot saved to {plot_save_path}")
    print(f"Metrics saved to {metrics_save_path}")
    print(f"TensorBoard logs saved to {tensorboard_dir}")
    print(f"Final metrics: {metrics}")

if __name__ == "__main__":
    main()