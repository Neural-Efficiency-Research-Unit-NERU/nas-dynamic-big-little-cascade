#!/usr/bin/env python
"""Generate confidence-vs-accuracy plots for CIFAR-10 ablation runs.

The plots are reliability-style diagrams: confidence bins on the x-axis and
observed little-model accuracy on the y-axis. They use the saved retrained
checkpoints, so they reflect the actual little-model confidence distribution
rather than only the scalar ECE values saved in CSV files.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_data_loaders
from src.models.paired_models import CascadePair
from src.nas.search_space import PairGenotype
from src.training.utils import collect_confidence_data, compute_ece, get_device, set_seed


ABLATION_COLORS = {
    "C1": "#1f77b4",
    "C2": "#ff7f0e",
    "C3": "#9467bd",
    "C4": "#2ca02c",
}

VARIANTS = {
    "C1": ("CE only", "cifar10_A_retrain_ce_only", ABLATION_COLORS["C1"]),
    "C2": ("CE + smoothing", "cifar10_A_retrain_ce_smooth", ABLATION_COLORS["C2"]),
    "C3": ("Cascade loss", "cifar10_A_retrain_cascade_only", ABLATION_COLORS["C3"]),
    "C4": ("Cascade + smoothing", "cifar10_A_retrain_cascade_smooth", ABLATION_COLORS["C4"]),
}


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.png / .pdf")


def _load_genotype(summary_path: Path) -> PairGenotype:
    with summary_path.open() as f:
        for line in f:
            if line.startswith("Genotype:"):
                genotype_dict = ast.literal_eval(line.split("Genotype:", 1)[1].strip())
                return PairGenotype.from_dict(genotype_dict)
    raise ValueError(f"No genotype found in {summary_path}")


def _load_model(exp_dir: Path, arch_idx: int, device: torch.device) -> CascadePair:
    genotype = _load_genotype(exp_dir / f"nas_arch_{arch_idx}_summary.txt")
    model = CascadePair(genotype)
    checkpoint = exp_dir / f"nas_arch_{arch_idx}_best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def _collect_variant_confidence(
    exp_dir: Path,
    loader,
    device: torch.device,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    confidences = []
    correctness = []
    for arch_idx in range(1, top_k + 1):
        checkpoint = exp_dir / f"nas_arch_{arch_idx}_best.pt"
        summary = exp_dir / f"nas_arch_{arch_idx}_summary.txt"
        if not checkpoint.exists() or not summary.exists():
            print(f"Skipping {exp_dir.name} arch {arch_idx}: checkpoint or summary missing")
            continue

        model = _load_model(exp_dir, arch_idx, device)
        conf, corr = collect_confidence_data(model, loader, device, temperature=1.0)
        confidences.append(conf)
        correctness.append(corr)

    if not confidences:
        return np.array([]), np.array([])
    return np.concatenate(confidences), np.concatenate(correctness)


def _bin_confidence_accuracy(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    mean_conf = np.full(n_bins, np.nan)
    accuracy = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins)

    for idx in range(n_bins):
        lo, hi = boundaries[idx], boundaries[idx + 1]
        if idx == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.any():
            mean_conf[idx] = float(confidences[mask].mean())
            accuracy[idx] = float(correctness[mask].mean())
            counts[idx] = int(mask.sum())

    return mean_conf, accuracy, counts


def plot_confidence_accuracy_overlay(
    variant_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    n_bins: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

    for key, (label, _, color) in VARIANTS.items():
        confidences, correctness = variant_data.get(key, (np.array([]), np.array([])))
        if len(confidences) == 0:
            continue
        mean_conf, accuracy, _ = _bin_confidence_accuracy(confidences, correctness, n_bins)
        valid = np.isfinite(mean_conf) & np.isfinite(accuracy)
        ece = compute_ece(confidences, correctness, n_bins=n_bins)
        ax.plot(
            mean_conf[valid],
            accuracy[valid],
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"{key}: {label} (ECE={ece:.3f})",
        )

    ax.set_xlabel("Mean Confidence")
    ax.set_ylabel("Observed Accuracy")
    ax.set_title("Confidence vs Accuracy by Ablation")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, output_dir / "routing", "confidence_accuracy_ablation_overlay")


def plot_confidence_accuracy_grid(
    variant_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    n_bins: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    bin_width = 1.0 / n_bins
    centers = np.linspace(bin_width / 2, 1 - bin_width / 2, n_bins)

    for ax, key in zip(axes.ravel(), VARIANTS):
        label, _, color = VARIANTS[key]
        confidences, correctness = variant_data.get(key, (np.array([]), np.array([])))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        if len(confidences) == 0:
            ax.set_title(f"{key}: {label} (missing)")
            continue

        mean_conf, accuracy, counts = _bin_confidence_accuracy(confidences, correctness, n_bins)
        valid = np.isfinite(accuracy)
        ece = compute_ece(confidences, correctness, n_bins=n_bins)
        ax.bar(
            centers[valid],
            accuracy[valid],
            width=bin_width * 0.9,
            color=color,
            alpha=0.72,
            edgecolor="black",
            linewidth=0.45,
            label="Observed accuracy",
        )

        for idx in np.where(valid)[0]:
            lo = min(mean_conf[idx], accuracy[idx])
            hi = max(mean_conf[idx], accuracy[idx])
            ax.bar(
                centers[idx],
                hi - lo,
                bottom=lo,
                width=bin_width * 0.9,
                color="red",
                alpha=0.25,
                edgecolor="none",
            )

        ax.set_title(f"{key}: {label} | ECE={ece:.3f} | n={int(counts.sum()):,}")
        ax.grid(alpha=0.22)

    for ax in axes[-1, :]:
        ax.set_xlabel("Confidence")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed Accuracy")

    fig.suptitle("Confidence vs Accuracy Reliability Diagrams", y=0.98)
    _save(fig, output_dir / "routing", "confidence_accuracy_ablation_grid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate confidence-vs-accuracy plots.")
    parser.add_argument("--experiments-root", default="experiments", type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data-root", default=None, type=Path)
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument("--n-bins", default=15, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    output_dir = args.output_dir or args.experiments_root / "plots"
    data_root = args.data_root or args.experiments_root.parent / "data"

    print(f"Using device: {device}")
    _, _, test_loader = get_data_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=str(data_root),
    )

    variant_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, (label, dirname, _) in VARIANTS.items():
        exp_dir = args.experiments_root / dirname
        if not exp_dir.exists():
            print(f"Skipping {key}: {exp_dir} missing")
            continue
        print(f"Collecting confidences for {key}: {label}")
        variant_data[key] = _collect_variant_confidence(exp_dir, test_loader, device, args.top_k)

    plot_confidence_accuracy_overlay(variant_data, output_dir, args.n_bins)
    plot_confidence_accuracy_grid(variant_data, output_dir, args.n_bins)


if __name__ == "__main__":
    main()
