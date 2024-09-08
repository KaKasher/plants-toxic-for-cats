import torch
import matplotlib.pyplot as plt
import json
from pathlib import Path

from model import setup_model
from data import setup_data, get_species_names
from train import train_test_loop
from metrics import calculate_metrics

def save_losses(train_losses, test_losses, filename):
    loss_data = {
        "train_losses": train_losses,
        "test_losses": test_losses
    }
    with open(filename, "w") as f:
        json.dump(loss_data, f)

def plot_losses(train_losses, test_losses, num_epochs, filename):
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
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = Path("../../data/train_test")
    train_path = data_path / "train"
    species_names = get_species_names(train_path)
    num_classes = len(species_names)

    plantnet_model = setup_model(num_classes)
    train_loader, test_loader, _ = setup_data()

    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 15

    train_losses, test_losses = train_test_loop(plantnet_model, train_loader, test_loader, criterion, device,
                                                num_epochs)

    # Save the model
    model_folder = Path("../../models")

    model_save_path = model_folder / "plantnet_finetuned_resnet34_v5.tar"
    torch.save(plantnet_model.state_dict(), model_save_path)

    # Save the losses
    loss_save_path = model_folder / "plantnet_finetuned_resnet34_v5_losses.json"
    save_losses(train_losses, test_losses, loss_save_path)

    # Plot and save the graph
    plot_save_path = model_folder / "plantnet_finetuned_resnet34_v5_losses_plot.png"
    plot_losses(train_losses, test_losses, num_epochs, plot_save_path)

    # Calculate and save metrics
    metrics = calculate_metrics(plantnet_model, test_loader, device)
    metrics_save_path = model_folder / "/models/plantnet_finetuned_resnet34_v5_metrics.json"
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f)

    print(f"Model saved to {model_save_path}")
    print(f"Loss data saved to {loss_save_path}")
    print(f"Loss plot saved to {plot_save_path}")
    print(f"Metrics saved to {metrics_save_path}")
    print(f"Final metrics: {metrics}")

if __name__ == "__main__":
    main()