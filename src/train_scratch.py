import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from src.utils import ensure_dir, set_seed


def build_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=str(Path(data_dir) / "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(Path(data_dir) / "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_dataset.class_to_idx


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if not name.startswith("layer4") and not name.startswith("fc"):
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_history_plot(history, output_path: str):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Loss vs Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[0, 1].plot(epochs, history["val_acc"], label="Val Accuracy")
    axes[0, 1].set_title("Accuracy vs Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[1, 0].set_title("Train Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")

    axes[1, 1].plot(epochs, history["val_acc"], label="Val Accuracy")
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_model(model, train_loader, val_loader, device, epochs: int = 20, lr: float = 3e-4, patience: int = 3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    print("Trainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}")
    print("\nFrozen layers:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"  - {name}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"[Partial Fine-Tuning] Epoch {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_total += loss.item() * images.size(0)
                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 with partial fine-tuning.")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_to_idx = build_dataloaders(args.data_dir, batch_size=args.batch_size)
    model = build_model(num_classes=len(class_to_idx))
    model.to(device)

    start = time.time()
    history = train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, patience=args.patience)
    training_time = time.time() - start

    ensure_dir("models")
    ensure_dir("results")
    checkpoint_path = Path("models") / "resnet18_partial_finetuning.pth"
    torch.save(model.state_dict(), checkpoint_path)

    history_payload = {
        "model": "resnet18_partial_finetuning",
        "class_to_idx": class_to_idx,
        "training_time_seconds": float(training_time),
        "history": history,
    }
    with open(Path("models") / "resnet18_partial_finetuning_history.json", "w", encoding="utf-8") as fh:
        json.dump(history_payload, fh, indent=2)

    save_history_plot(history, str(Path("results") / "partial_finetuning_history.png"))

    print(f"Model saved to: {checkpoint_path}")
    print(f"Training history saved to: models/resnet18_partial_finetuning_history.json")
    print(f"Training plot saved to: results/partial_finetuning_history.png")
    print(f"Training time: {training_time:.2f}s")


if __name__ == "__main__":
    main()
