import torch
from torchvision import models, transforms
from PIL import Image
import argparse
from pathlib import Path
import json
import timm

def classify_image(image_path):
    model_path = Path('../models/vit_b16_224_25e_256bs_0.001lr_adamW_transforms.tar')

    # Load the image
    image = Image.open(image_path)

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Preprocess the image
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    # Load the model
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=47)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()

    with torch.inference_mode():
        output = model(input_batch)

    # Load class labels
    with open('idx_to_class.json', 'r') as f:
        data = f.read()
    idx_to_class = json.loads(data)
    # Get probabilities
    probs = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 3 predictions
    top3_prob, top3_catid = torch.topk(probs, 3)

    # Return results
    results = []
    for i in range(3):
        results.append({
            'category': idx_to_class[str(top3_catid[i].item())],
            'probability': top3_prob[i].item()
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Classify an image using a PyTorch model.')
    parser.add_argument('image_path', type=str, help='Path to the input image')

    args = parser.parse_args()

    predictions = classify_image(args.image_path)

    for pred in predictions:
        print(f"Category: {pred['category']}, Probability: {pred['probability']:.4f}")


if __name__ == '__main__':
    main()