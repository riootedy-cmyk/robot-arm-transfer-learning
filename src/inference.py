import argparse
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

from src.utils import ensure_dir


def build_eval_loader(data_dir: str, batch_size: int = 32):
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(root=str(Path(data_dir) / "val"), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return val_loader, val_dataset.class_to_idx


def load_model(model_type: str, num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_confusion_matrix(cm, labels, output_path: str):
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
            value = cm[i, j]
            color = "white" if value > thresh else "black"
            plt.text(j, i, format(value, "d"), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_model(model_path: str, data_dir: str, model_type: str, device_name: str = "cuda"):
    device = torch.device(device_name)
    val_loader, class_to_idx = build_eval_loader(data_dir)
    model = load_model(model_type, num_classes=len(class_to_idx))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    start = time.time()

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Evaluating {model_type}"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, preds = torch.max(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    inference_latency = (time.time() - start) / max(len(y_true), 1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    label_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    ensure_dir("results")
    save_confusion_matrix(cm, label_names, str(Path("results") / f"{model_type}_confusion_matrix.png"))

    history_path = Path("models") / f"resnet18_{model_type}_history.json"
    training_time = 0.0
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        training_time = float(payload.get("training_time_seconds", 0.0))

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

    metrics_path = Path("results") / f"{model_type}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--model-type", type=str, required=True, choices=["feature_extraction", "partial_finetuning", "scratch"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_dir, args.model_type, args.device)


if __name__ == "__main__":
    main()
