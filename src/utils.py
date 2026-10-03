import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


LABELS = ["red", "green", "blue"]


def load_model(model_type: str):
    checkpoint_name = {
        "feature_extraction": "resnet18_feature_extraction.pth",
        "partial_finetuning": "resnet18_partial_finetuning.pth",
        "scratch": "resnet18_scratch.pth",
    }[model_type]
    checkpoint_path = Path("models") / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}. Train the model first.")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def preprocess_image(image_path: str):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model", type=str, required=True, choices=["feature_extraction", "partial_finetuning", "scratch"], help="Model type")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model(args.model)
    input_tensor = preprocess_image(str(image_path))

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, idx = torch.max(probs, dim=1)

    predicted = LABELS[idx.item()]
    conf_percent = confidence.item() * 100.0
    print(f"Predicted class: {predicted.upper()}")
    print(f"Confidence: {conf_percent:.2f}%")


if __name__ == "__main__":
    main()
