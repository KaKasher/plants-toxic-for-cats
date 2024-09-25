import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np


def calculate_metrics(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_top3 = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            _, top_3_indices = outputs.topk(3, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_top3.extend(top_3_indices.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_top3 = np.array(all_top3)

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    top3_acc = np.mean([label in pred for label, pred in zip(all_labels, all_top3)])

    return {
        'balanced_accuracy': balanced_acc,
        'weighted_f1': weighted_f1,
        'top3_accuracy': top3_acc
    }