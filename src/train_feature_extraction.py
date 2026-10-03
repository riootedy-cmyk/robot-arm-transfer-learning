import argparse
import json
import random
from pathlib import Path

from src.utils import CLASS_NAMES, ensure_dir, get_image_files, set_seed


def copy_dataset(raw_root: str, output_root: str, split_ratio: float = 0.8, seed: int = 42):
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    set_seed(seed)

    before_summary = {}
    for class_name in CLASS_NAMES:
        class_dir = raw_root / class_name
        files = get_image_files(class_dir)
        before_summary[class_name] = len(files)
        if len(files) == 0:
            raise FileNotFoundError(f"No images found in {class_dir}. Please add images to dataset_raw/{class_name}.")

    for class_name in CLASS_NAMES:
        src_dir = raw_root / class_name
        train_dir = output_root / "train" / class_name
        val_dir = output_root / "val" / class_name
        ensure_dir(train_dir)
        ensure_dir(val_dir)

        files = get_image_files(src_dir)
        random.Random(seed).shuffle(files)

        if len(files) < 2:
            raise ValueError(f"Class '{class_name}' has only {len(files)} image(s). At least 2 images are needed for train/validation split.")

        split_index = max(1, int(len(files) * split_ratio))
        train_files = files[:split_index]
        val_files = files[split_index:]

        if not val_files:
            train_files = files[:-1]
            val_files = files[-1:]

        for src in train_files:
            dest = train_dir / src.name
            dest.write_bytes(src.read_bytes())

        for src in val_files:
            dest = val_dir / src.name
            dest.write_bytes(src.read_bytes())

    after_summary = {"train": {}, "val": {}}
    for split_name in ["train", "val"]:
        for class_name in CLASS_NAMES:
            after_summary[split_name][class_name] = len(get_image_files(output_root / split_name / class_name))

    print("Dataset summary before split:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {before_summary[class_name]}")

    print("Dataset summary after split:")
    for split_name, counts in after_summary.items():
        print(f"  {split_name}:")
        for class_name in CLASS_NAMES:
            print(f"    {class_name}: {counts[class_name]}")

    summary = {"before": before_summary, "after": after_summary}
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return before_summary, after_summary


def main():
    parser = argparse.ArgumentParser(description="Split raw dataset into train/validation folders")
    parser.add_argument("--raw-root", type=str, default="dataset_raw", help="Path to raw dataset root")
    parser.add_argument("--output-root", type=str, default="dataset", help="Path to processed dataset root")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train ratio for the split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)

    for class_name in CLASS_NAMES:
        if not (raw_root / class_name).exists():
            raise FileNotFoundError(f"Folder not found: {raw_root / class_name}. Please create dataset_raw/{class_name}.")

    copy_dataset(str(raw_root), str(output_root), split_ratio=args.split_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
