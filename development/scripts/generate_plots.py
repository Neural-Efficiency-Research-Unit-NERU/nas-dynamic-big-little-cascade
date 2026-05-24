#!/usr/bin/env python
"""Generate standard visualizations from CIFAR-10 experiment results."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization.plots import (
    load_training_history,
    plot_nas_search_progress,
    plot_training_curve_overlay,
    plot_training_curves,
)

ABLATION_COLORS = {
    "C1": "#1f77b4",
    "C2": "#ff7f0e",
    "C3": "#9467bd",
    "C4": "#2ca02c",
}


def generate_standard(exp_base: Path, output_dir: Path, dataset: str):
    """Generate NAS search progress and per-architecture training curves."""
    nas_retrain_dir = exp_base / f"{dataset}_A_retrain_cascade_smooth"
    if nas_retrain_dir.exists():
        for i in range(1, 10):
            history_path = nas_retrain_dir / f"nas_arch_{i}_history.csv"
            if history_path.exists():
                print(f"Plotting NAS Arch {i} training curves from {nas_retrain_dir.name}...")
                history = load_training_history(history_path)
                plot_training_curves(history, f"NAS Arch {i}", output_dir / "training")

    search_dir = exp_base / f"{dataset}_cosearch_nas"
    if search_dir.exists():
        search_log = search_dir / "search_log.csv"
        if search_log.exists():
            print("Plotting NAS search progress...")
            plot_nas_search_progress(search_log, output_dir / "search")


def generate_rq2(exp_base: Path, output_dir: Path, dataset: str):
    """Generate C1-C4 training-curve overlay from retrained histories."""
    variant_dir_suffix = {
        "C1": "ce_only",
        "C2": "ce_smooth",
        "C3": "cascade_only",
        "C4": "cascade_smooth",
    }

    histories = []
    for variant, color in ABLATION_COLORS.items():
        suffix = variant_dir_suffix[variant]
        retrain_dir = exp_base / f"{dataset}_A_retrain_{suffix}"
        history_path = retrain_dir / "nas_arch_1_history.csv"
        if not history_path.exists():
            print(f"  Warning: no history found for {variant} at {history_path}")
            continue

        print(f"  Loading {variant} training history from {history_path}")
        raw = load_training_history(history_path)
        if "val_cascade_acc" not in raw:
            print(f"  Warning: {variant} history missing val_cascade_acc, skipping")
            continue

        histories.append({
            "name": variant,
            "epochs": raw["epoch"],
            "val_cascade_acc": raw["val_cascade_acc"],
            "color": color,
        })

    if not histories:
        print("  No C1-C4 histories found, skipping training curve overlay.")
        return

    print("Plotting C1-C4 training curve overlay...")
    plot_training_curve_overlay(histories, output_dir=output_dir / "training")


def main():
    parser = argparse.ArgumentParser(
        description="Generate standard visualizations from CIFAR-10 experiment results."
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="Dataset name used to construct experiment directory names. Default: cifar10",
    )
    parser.add_argument(
        "--mode",
        default="standard",
        choices=["standard", "rq2", "all"],
        help="Which plots to generate. Default: standard",
    )
    args = parser.parse_args()

    exp_base = Path(__file__).resolve().parent.parent / "experiments"
    output_dir = exp_base / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("standard", "all"):
        print(f"=== Standard plots (dataset={args.dataset}) ===")
        generate_standard(exp_base, output_dir, args.dataset)

    if args.mode in ("rq2", "all"):
        print(f"\n=== C1-C4 training comparison (dataset={args.dataset}) ===")
        generate_rq2(exp_base, output_dir, args.dataset)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
