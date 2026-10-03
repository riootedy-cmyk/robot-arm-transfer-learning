import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command):
    print(f"\n>>> {command}")
    result = subprocess.run(command, shell=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")


def main():
    parser = argparse.ArgumentParser(description="Run dataset prep, training, evaluation, and inference for the color classification project.")
    parser.add_argument("--prepare", action="store_true", help="Prepare train/val split from dataset_raw")
    parser.add_argument("--train-feature", action="store_true", help="Train feature extraction model")
    parser.add_argument("--train-finetune", action="store_true", help="Train partial fine-tuning model")
    parser.add_argument("--train-scratch", action="store_true", help="Train scratch model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate all available model checkpoints")
    parser.add_argument("--compare", action="store_true", help="Compare evaluation JSON results into a CSV table")
    parser.add_argument("--infer", type=str, default=None, help="Run inference on a single image file")
    parser.add_argument("--image", type=str, default=None, help="Path to image for inference")
    parser.add_argument("--model", type=str, default="feature_extraction", choices=["feature_extraction", "partial_finetuning", "scratch"], help="Model type for inference or evaluation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    python_exe = sys.executable

    if args.prepare:
        run_command(f'"{python_exe}" "{project_root / "src" / "prepare_dataset.py"}" --raw-root dataset_raw --output-root dataset --split-ratio 0.8')

    if args.train_feature:
        run_command(f'"{python_exe}" "{project_root / "src" / "train_feature_extraction.py"}" --data-dir dataset --epochs 15 --batch-size 32 --lr 1e-3 --seed 42')

    if args.train_finetune:
        run_command(f'"{python_exe}" "{project_root / "src" / "train_finetuning.py"}" --data-dir dataset --epochs 20 --batch-size 32 --lr 3e-4 --seed 42')

    if args.train_scratch:
        run_command(f'"{python_exe}" "{project_root / "src" / "train_scratch.py"}" --data-dir dataset --epochs 25 --batch-size 32 --lr 1e-3 --seed 42')

    if args.evaluate:
        for model_type in ["feature_extraction", "partial_finetuning", "scratch"]:
            checkpoint = project_root / "models" / f"resnet18_{model_type}.pth"
            if checkpoint.exists():
                run_command(f'"{python_exe}" "{project_root / "src" / "evaluate.py"}" --model-path "{checkpoint}" --data-dir dataset --model-type {model_type}')
            else:
                print(f"Skipping evaluation for {model_type}: checkpoint not found at {checkpoint}")

    if args.compare:
        import json
        import csv

        results_dir = project_root / "results"
        metrics_files = sorted(results_dir.glob("*_metrics.json"))
        if not metrics_files:
            print("No *_metrics.json results found. Train and evaluate a model first.")
        else:
            rows = []
            for path in metrics_files:
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                rows.append({
                    "Method": path.stem.replace("_metrics", "").replace("_", " ").title(),
                    "Accuracy": payload.get("accuracy", 0.0),
                    "Precision": payload.get("precision", 0.0),
                    "Recall": payload.get("recall", 0.0),
                    "F1": payload.get("f1", 0.0),
                    "Training Time": payload.get("training_time_seconds", 0.0),
                    "Inference Latency": payload.get("inference_latency_seconds", 0.0),
                })

            output_csv = results_dir / "comparison_table.csv"
            with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["Method", "Accuracy", "Precision", "Recall", "F1", "Training Time", "Inference Latency"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"Comparison table saved to {output_csv}")

    if args.infer is not None:
        if args.image is None:
            raise ValueError("--image is required when using --infer")
        run_command(f'"{python_exe}" "{project_root / "src" / "inference.py"}" --image "{args.image}" --model {args.model}')


if __name__ == "__main__":
    main()
