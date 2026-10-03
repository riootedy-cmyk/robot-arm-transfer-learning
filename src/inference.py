import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


MODEL_MAP = {
    "feature_extraction": "resnet18_feature_extraction.pth",
    "partial_finetuning": "resnet18_partial_finetuning.pth",
    "scratch": "resnet18_scratch.pth",
}


def build_val_loader(data_dir, batch_size=32):
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(root=str(Path(data_dir) / "val"), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return val_loader, val_dataset.class_to_idx


def build_model(model_type, num_classes):
    if model_type == "feature_extraction":
        model = models.resnet18(weights=None)
    elif model_type == "partial_finetuning":
        model = models.resnet18(weights=None)
    elif model_type == "scratch":
        model = models.resnet18(weights=None)
    else:
        raise ValueError(f"Unrecognized model type: {model_type}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_confusion_matrix(cm, labels, output_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_model(model_path, data_dir, model_type, device_name="cpu"):
    device = torch.device(device_name)
    val_loader, class_to_idx = build_val_loader(data_dir)
    model = build_model(model_type, num_classes=len(class_to_idx))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    start = time.time()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Evaluating {model_type}"):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            _, preds = torch.max(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    inference_latency = (time.time() - start) / max(len(y_true), 1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    label_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(cm, label_names, output_dir / f"{model_type}_confusion_matrix.png")

    training_history_path = Path("models") / f"resnet18_{model_type}_history.json"
    training_time = 0.0
    if training_history_path.exists():
        with open(training_history_path, "r", encoding="utf-8") as fh:
            history_payload = json.load(fh)
        training_time = history_payload.get("training_time_seconds", 0.0)

    metrics = {
        "model": model_type,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "class_names": label_names,
        "training_time_seconds": float(training_time),
        "inference_latency_seconds": float(inference_latency),
    }

    with open(output_dir / f"{model_type}_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet-18 color classification models.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--model-type", type=str, choices=["feature_extraction", "partial_finetuning", "scratch"], required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_dir, args.model_type, args.device)


if __name__ == "__main__":
    main()
