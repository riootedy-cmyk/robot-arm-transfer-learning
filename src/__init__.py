import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_command(command: str):
    print(f"\n>>> {command}")
    result = subprocess.run(command, shell=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")


def main():
    parser = argparse.ArgumentParser(description="Run training and evaluation for the color-classification project.")
    parser.add_argument("--prepare", action="store_true", help="Split dataset_raw into train/val")
    parser.add_argument("--train-feature", action="store_true", help="Train feature extraction model")
    parser.add_argument("--train-finetune", action="store_true", help="Train partial fine-tuning model")
    parser.add_argument("--train-scratch", action="store_true", help="Train scratch model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate all available checkpoints")
    parser.add_argument("--compare", action="store_true", help="Create a comparison table from evaluation JSON files")
    parser.add_argument("--infer", type=str, default=None, help="Run inference on a single image")
    parser.add_argument("--image", type=str, default=None, help="Image path for inference")
    parser.add_argument("--model", type=str, default="feature_extraction", choices=["feature_extraction", "partial_finetuning", "scratch"], help="Model type")
    args = parser.parse_args()

    py = sys.executable

    if args.prepare:
        run_command(f'"{py}" "{ROOT / "src" / "prepare_dataset.py"}" --raw-root dataset_raw --output-root dataset --split-ratio 0.8 --seed 42')

    if args.train_feature:
        run_command(f'"{py}" "{ROOT / "src" / "train_feature_extraction.py"}" --data-dir dataset --epochs 15 --batch-size 32 --lr 1e-3 --seed 42')

    if args.train_finetune:
        run_command(f'"{py}" "{ROOT / "src" / "train_finetuning.py"}" --data-dir dataset --epochs 20 --batch-size 32 --lr 3e-4 --seed 42')

    if args.train_scratch:
        run_command(f'"{py}" "{ROOT / "src" / "train_scratch.py"}" --data-dir dataset --epochs 25 --batch-size 32 --lr 1e-3 --seed 42')

    if args.evaluate:
        for model_type in ["feature_extraction", "partial_finetuning", "scratch"]:
            checkpoint = ROOT / "models" / f"resnet18_{model_type}.pth"
            if checkpoint.exists():
                run_command(f'"{py}" "{ROOT / "src" / "evaluate.py"}" --model-path "{checkpoint}" --data-dir dataset --model-type {model_type}')
            else:
                print(f"Skipping {model_type}: checkpoint not found at {checkpoint}")

    if args.compare:
        results_dir = ROOT / "results"
        metrics_files = sorted(results_dir.glob("*_metrics.json"))
        if not metrics_files:
            print("No metrics JSON files found. Train and evaluate models first.")
        else:
            rows = []
            for path in metrics_files:
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                method = path.stem.replace("_metrics", "").replace("_", " ").title()
                rows.append({
                    "Method": method,
                    "Accuracy": payload.get("accuracy", 0.0),
                    "Precision": payload.get("precision", 0.0),
                    "Recall": payload.get("recall", 0.0),
                    "F1": payload.get("f1", 0.0),
                    "Training Time": payload.get("training_time_seconds", 0.0),
                    "Inference Latency": payload.get("inference_latency_seconds", 0.0),
                })

            out_csv = results_dir / "comparison_table.csv"
            with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["Method", "Accuracy", "Precision", "Recall", "F1", "Training Time", "Inference Latency"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"Comparison table saved to {out_csv}")

    if args.infer is not None:
        if args.image is None:
            raise ValueError("--image is required when using --infer")
        run_command(f'"{py}" "{ROOT / "src" / "inference.py"}" --image "{args.image}" --model {args.model}')


if __name__ == "__main__":
    main()
