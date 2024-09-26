import gradio as gr
from pathlib import Path
from torchvision import transforms
import timm
import torch
import json
import os

# Load toxicity data
def load_toxicity_data(file_path='combined_plant_toxicity.json'):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Toxicity information will not be available.")
        return {}

# Load class labels
def load_class_labels(file_path='idx_to_class.json'):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Class labels will not be available.")
        return {}




# Global variables
toxicity_data = load_toxicity_data()
idx_to_class = load_class_labels()
model_path = Path('vit_b16_224_25e_256bs_0.001lr_adamW_transforms.tar')
model = None

# Load model
try:
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=47)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")


# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def classify_image(input_image):
    if input_image is None:
        return None, "Error: No image uploaded. Please choose an image and try again."

    if model is None:
        return None, "Error: Model could not be loaded. Please check the model path and try again."

    try:
        # Preprocess the image
        input_tensor = transform(input_image).unsqueeze(0)

        with torch.inference_mode():
            output = model(input_tensor)
            predictions = torch.nn.functional.softmax(output[0], dim=0)
            confidences = {idx_to_class[str(i)]: float(predictions[i]) for i in range(47)}

        # Sort confidences and get top 3
        top_3 = sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:3]

        # Prepare the output for Label
        label_output = {plant: conf for plant, conf in top_3}

        # Prepare the toxicity information
        toxicity_info = "### Toxicity Information\n\n"
        for plant, _ in top_3:
            toxicity = toxicity_data.get(plant, "Unknown")
            toxicity_info += f"- **{plant}**: {toxicity}\n"

        return label_output, toxicity_info

    except Exception as e:
        return None, f"An error occurred during image classification: {str(e)}"

demo = gr.Interface(
    classify_image,
    gr.Image(type="pil"),
    [
        gr.Label(num_top_classes=3, label="Top 3 Predictions"),
        gr.Markdown(label="Toxicity Information")
    ],
    title="🌱 Cat-Safe Plant Classifier 🐱",
    description="Upload an image of a plant to get its classification and toxicity information for cats. Includes 47 most popular house plants",
    examples=[["examples/" + example] for example in os.listdir("examples")]
)

if __name__ == "__main__":
    demo.launch()