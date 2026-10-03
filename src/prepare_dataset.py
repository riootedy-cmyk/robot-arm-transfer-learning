import argparse
import json
import random
from pathlib import Path

from PIL import Image


def safe_count_images(directory):
    if not directory.exists():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_image_files(directory):
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def split_dataset(raw_root, output_root, split_ratio=0.8, seed=42):
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    class_names = ["red", "green", "blue"]

    summary_before = {}
    for class_name in class_names:
        class_dir = raw_root / class_name
        files = get_image_files(class_dir)
        summary_before[class_name] = len(files)
        if len(files) == 0:
            raise FileNotFoundError(f"No images found in {class_dir}. Please place images in dataset_raw/{class_name}.")

    for class_name in class_names:
        source_dir = raw_root / class_name
        train_dir = output_root / "train" / class_name
        val_dir = output_root / "val" / class_name
        ensure_dir(train_dir)
        ensure_dir(val_dir)

        files = get_image_files(source_dir)
        random.Random(seed).shuffle(files)

        if len(files) < 2:
            raise ValueError(f"Class '{class_name}' has only {len(files)} image(s). Need at least 2 images for train/validation split.")

        split = max(1, int(len(files) * split_ratio))
        train_files = files[:split]
        val_files = files[split:]
        if not val_files:
            # if split leaves no validation data, move the last image to validation
            train_files = files[:-1]
            val_files = files[-1:]

        for src in train_files:
            dest = train_dir / src.name
            if dest.exists():
                dest.unlink()
            dest.write_bytes(src.read_bytes())

        for src in val_files:
            dest = val_dir / src.name
            if dest.exists():
                dest.unlink()
            dest.write_bytes(src.read_bytes())

    summary_after = {}
    for split_name in ["train", "val"]:
        summary_after[split_name] = {}
        for class_name in class_names:
            summary_after[split_name][class_name] = safe_count_images(output_root / split_name / class_name)

    print("Dataset summary before split:")
    for class_name, count in summary_before.items():
        print(f"  {class_name}: {count}")
    print("Dataset summary after split:")
    for split_name, counts in summary_after.items():
        print(f"  {split_name}:")
        for class_name, count in counts.items():
            print(f"    {class_name}: {count}")

    return summary_before, summary_after


def print_class_counts(folder):
    for class_name in ["red", "green", "blue"]:
        count = safe_count_images(Path(folder) / class_name)
        print(f"{class_name}: {count}")
