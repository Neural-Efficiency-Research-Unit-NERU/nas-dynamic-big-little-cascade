#!/usr/bin/env python
"""Generate additional thesis-oriented plots from saved CIFAR-10 results.

These plots are intentionally CSV-only: they use the threshold sweeps and
independent cascade results already produced by the evaluation pipeline.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ABLATION_COLORS = {
    "C1": "#1f77b4",  # blue
    "C2": "#ff7f0e",  # orange
    "C3": "#9467bd",  # purple
    "C4": "#2ca02c",  # green
}
INDEPENDENT_COLOR = "#7f7f7f"

VARIANTS = {
    "C1": ("CE only", "cifar10_A_retrain_ce_only", ABLATION_COLORS["C1"]),
    "C2": ("CE + smoothing", "cifar10_A_retrain_ce_smooth", ABLATION_COLORS["C2"]),
    "C3": ("Cascade loss", "cifar10_A_retrain_cascade_only", ABLATION_COLORS["C3"]),
    "C4": ("Cascade + smoothing", "cifar10_A_retrain_cascade_smooth", ABLATION_COLORS["C4"]),
}
FIXED_THRESHOLD = 0.70


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"approach", "arch_id", "genotype"}:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.png / .pdf")


def _fixed_rows(rows: list[dict], threshold: float = FIXED_THRESHOLD) -> list[dict]:
    return [
        row for row in rows
        if "threshold" in row and abs(float(row["threshold"]) - threshold) < 1e-9
    ]


def _zoom_axis_to_data(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    x_pad_fraction: float = 0.08,
    y_pad_fraction: float = 0.12,
) -> None:
    if not xs or not ys:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, max(abs(x_max), 1.0) * 0.05)
    y_span = max(y_max - y_min, max(abs(y_max), 1.0) * 0.02)
    ax.set_xlim(x_min - x_span * x_pad_fraction, x_max + x_span * x_pad_fraction)
    ax.set_ylim(y_min - y_span * y_pad_fraction, y_max + y_span * y_pad_fraction)


def _zoom_bar_y_axis(
    ax: plt.Axes,
    values: list[float],
    lower_pad: float = 0.8,
    upper_pad: float = 1.2,
) -> None:
    if not values:
        return
    y_min = min(values)
    y_max = max(values)
    ax.set_ylim(max(0.0, y_min - lower_pad), min(100.0, y_max + upper_pad))


def _load_variant_sweeps(exp_base: Path) -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {}
    for key, (_, dirname, _) in VARIANTS.items():
        rows = []
        for path in sorted((exp_base / dirname).glob("nas_arch_*_threshold_sweep.csv")):
            if "tempscaled" in path.name:
                continue
            arch_id = path.stem.split("_threshold_sweep")[0]
            for row in _read_csv(path):
                row["arch_id"] = arch_id
                rows.append(row)
        data[key] = rows
    return data


def _mean_by_threshold(rows: list[dict]) -> dict[float, dict]:
    groups: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        groups[float(row["threshold"])].append(row)

    means = {}
    for threshold, items in groups.items():
        means[threshold] = {
            key: float(np.mean([r[key] for r in items]))
            for key in [
                "cascade_acc",
                "exit_ratio",
                "cascade_flops",
                "little_ece",
                "routing_error_rate",
            ]
            if key in items[0]
        }
    return dict(sorted(means.items()))


def _colors_for(labels: list[str]) -> list[str]:
    return [ABLATION_COLORS.get(label, "gray") for label in labels]


def plot_ablation_accuracy_bars(variant_rows: dict[str, list[dict]], output_dir: Path) -> None:
    labels = []
    fixed_means = []

    for key, rows in variant_rows.items():
        by_arch: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_arch[row["arch_id"]].append(row)
        fixed = [
            next(r for r in items if abs(r["threshold"] - FIXED_THRESHOLD) < 1e-9)["cascade_acc"]
            for items in by_arch.values()
        ]
        labels.append(key)
        fixed_means.append(np.mean(fixed) * 100)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x, fixed_means, 0.55, color=_colors_for(labels), edgecolor="black", linewidth=0.4)
    for xpos, value in zip(x, fixed_means):
        ax.text(xpos, value + 0.15, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Cascade Accuracy (%)")
    ax.set_title("Ablation Accuracy Summary at Threshold 0.70")
    _zoom_bar_y_axis(ax, fixed_means)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir / "ablations", "ablation_accuracy_summary")


def plot_ablation_accuracy_bars_by_arch(
    variant_rows: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Plot threshold-0.70 ablation bars per architecture."""
    arch_ids = sorted({
        row["arch_id"]
        for rows in variant_rows.values()
        for row in rows
    })

    for arch_id in arch_ids:
        labels = []
        fixed_vals = []
        for key, rows in variant_rows.items():
            arch_rows = [r for r in rows if r["arch_id"] == arch_id]
            if not arch_rows:
                continue
            fixed = next((r for r in arch_rows if abs(r["threshold"] - FIXED_THRESHOLD) < 1e-9), None)
            if fixed is None:
                continue
            labels.append(key)
            fixed_vals.append(fixed["cascade_acc"] * 100)

        if not labels:
            continue

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(x, fixed_vals, 0.55, color=_colors_for(labels), edgecolor="black", linewidth=0.4)
        for xpos, value in zip(x, fixed_vals):
            ax.text(xpos, value + 0.15, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Cascade Accuracy (%)")
        title_arch = str(arch_id).replace("nas_arch_", "Architecture ")
        ax.set_title(f"Ablation Accuracy Summary at Threshold 0.70 - {title_arch}")
        _zoom_bar_y_axis(ax, fixed_vals)
        ax.grid(axis="y", alpha=0.25)
        arch_suffix = str(arch_id).replace("nas_arch_", "arch_")
        _save(fig, output_dir / "ablations", f"ablation_accuracy_summary_{arch_suffix}")


def plot_ablation_little_accuracy_bars(variant_rows: dict[str, list[dict]], output_dir: Path) -> None:
    labels = []
    fixed_means = []

    for key, rows in variant_rows.items():
        by_arch: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_arch[row["arch_id"]].append(row)
        fixed = []
        for items in by_arch.values():
            row = next((r for r in items if abs(r["threshold"] - FIXED_THRESHOLD) < 1e-9), None)
            if row is not None and "little_acc" in row:
                fixed.append(row["little_acc"])
        if not fixed:
            continue
        labels.append(key)
        fixed_means.append(np.mean(fixed) * 100)

    if not labels:
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x, fixed_means, 0.55, color=_colors_for(labels), edgecolor="black", linewidth=0.4)
    for xpos, value in zip(x, fixed_means):
        ax.text(xpos, value + 0.15, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Little Accuracy (%)")
    ax.set_title("Ablation Little-model Accuracy at Threshold 0.70")
    _zoom_bar_y_axis(ax, fixed_means)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir / "ablations", "ablation_little_accuracy_summary")


def plot_ablation_little_accuracy_bars_by_arch(
    variant_rows: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    arch_ids = sorted({
        row["arch_id"]
        for rows in variant_rows.values()
        for row in rows
    })

    for arch_id in arch_ids:
        labels = []
        fixed_vals = []
        for key, rows in variant_rows.items():
            arch_rows = [r for r in rows if r["arch_id"] == arch_id]
            if not arch_rows:
                continue
            row = next((r for r in arch_rows if abs(r["threshold"] - FIXED_THRESHOLD) < 1e-9), None)
            if row is None or "little_acc" not in row:
                continue
            labels.append(key)
            fixed_vals.append(row["little_acc"] * 100)

        if not labels:
            continue

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(x, fixed_vals, 0.55, color=_colors_for(labels), edgecolor="black", linewidth=0.4)
        for xpos, value in zip(x, fixed_vals):
            ax.text(xpos, value + 0.15, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Little Accuracy (%)")
        title_arch = str(arch_id).replace("nas_arch_", "Architecture ")
        ax.set_title(f"Ablation Little-model Accuracy at Threshold 0.70 - {title_arch}")
        _zoom_bar_y_axis(ax, fixed_vals)
        ax.grid(axis="y", alpha=0.25)
        arch_suffix = str(arch_id).replace("nas_arch_", "arch_")
        _save(fig, output_dir / "ablations", f"ablation_little_accuracy_summary_{arch_suffix}")


def plot_ablation_big_accuracy_bars(variant_rows: dict[str, list[dict]], output_dir: Path) -> None:
    labels = []
    fixed_means = []

    for key, rows in variant_rows.items():
        by_arch: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_arch[row["arch_id"]].append(row)
        fixed = []
        for items in by_arch.values():
            row = next((r for r in items if abs(r["threshold"] - FIXED_THRESHOLD) < 1e-9), None)
            if row is not None and "big_acc" in row:
                fixed.append(row["big_acc"])
        if not fixed:
            continue
        labels.append(key)
        fixed_means.append(np.mean(fixed) * 100)

    if not labels:
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x, fixed_means, 0.55, color=_colors_for(labels), edgecolor="black", linewidth=0.4)
    for xpos, value in zip(x, fixed_means):
        ax.text(xpos, value + 0.15, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Big Accuracy (%)")
    ax.set_title("Ablation Big-model Accuracy at Threshold 0.70")
    _zoom_bar_y_axis(ax, fixed_means)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir / "ablations", "ablation_big_accuracy_summary")


def plot_ablation_big_accuracy_bars_by_arch(
    variant_rows: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    arch_ids = sorted({
        row["arch_id"]
        for rows in variant_rows.values()
        for row in rows
    })

    for arch_id in arch_ids:
        labels = []
        fixed_vals = []
        for key, rows in variant_rows.items():
            arch_rows = [r for r in rows if r["arch_id"] == arch_id]
            if not arch_rows:
                continue
            row = next((r for r in arch_rows if abs(r["threshold"] - FIXED_THRESHOLD) < 1e-9), None)
            if row is None or "big_acc" not in row:
                continue
            labels.append(key)
            fixed_vals.append(row["big_acc"] * 100)

        if not labels:
            continue

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(x, fixed_vals, 0.55, color=_colors_for(labels), edgecolor="black", linewidth=0.4)
        for xpos, value in zip(x, fixed_vals):
            ax.text(xpos, value + 0.15, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Big Accuracy (%)")
        title_arch = str(arch_id).replace("nas_arch_", "Architecture ")
        ax.set_title(f"Ablation Big-model Accuracy at Threshold 0.70 - {title_arch}")
        _zoom_bar_y_axis(ax, fixed_vals)
        ax.grid(axis="y", alpha=0.25)
        arch_suffix = str(arch_id).replace("nas_arch_", "arch_")
        _save(fig, output_dir / "ablations", f"ablation_big_accuracy_summary_{arch_suffix}")


def plot_ablation_training_overlays(exp_base: Path, output_dir: Path) -> None:
    """Plot C1-C4 validation cascade-accuracy curves for each retrained arch."""
    arch_ids = sorted({
        int(path.stem.split("_")[2])
        for _, dirname, _ in VARIANTS.values()
        for path in (exp_base / dirname).glob("nas_arch_*_history.csv")
    })

    for arch_id in arch_ids:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False

        for key, (label, dirname, color) in VARIANTS.items():
            history_path = exp_base / dirname / f"nas_arch_{arch_id}_history.csv"
            if not history_path.exists():
                continue

            rows = _read_csv(history_path)
            if not rows or "val_cascade_acc" not in rows[0]:
                continue

            epochs = [r["epoch"] for r in rows]
            acc = [r["val_cascade_acc"] * 100 for r in rows]
            ax.plot(epochs, acc, color=color, linewidth=1.7, label=f"{key}: {label}")
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Cascade Accuracy (%)")
        ax.set_title(f"C1-C4 Training Curve Overlay - Architecture {arch_id}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        _save(fig, output_dir / "training", f"training_curve_overlay_arch_{arch_id}")


def plot_ablation_threshold_sensitivity(variant_rows: dict[str, list[dict]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = []
    acc = []
    exit_ratio = []
    for key, (label, _, color) in VARIANTS.items():
        rows = _fixed_rows(variant_rows[key])
        if not rows:
            continue
        labels.append(key)
        acc.append(np.mean([r["cascade_acc"] for r in rows]) * 100)
        exit_ratio.append(np.mean([r["exit_ratio"] for r in rows]))

    x = np.arange(len(labels))
    colors = [VARIANTS[key][2] for key in labels]
    axes[0].bar(x, acc, color=colors, edgecolor="black", linewidth=0.4)
    axes[1].bar(x, exit_ratio, color=colors, edgecolor="black", linewidth=0.4)
    for ax, vals in [(axes[0], acc), (axes[1], exit_ratio)]:
        for xpos, val in zip(x, vals):
            ax.text(xpos, val + (0.15 if ax is axes[0] else 0.01), f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    axes[0].set_xlabel("Ablation")
    axes[0].set_ylabel("Mean Cascade Accuracy (%)")
    axes[0].set_title("Accuracy at Threshold 0.70")
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("Ablation")
    axes[1].set_ylabel("Mean Exit Ratio")
    axes[1].set_title("Exit Ratio at Threshold 0.70")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    _save(fig, output_dir / "routing", "ablation_threshold_sensitivity")


def plot_ablation_threshold_sweep_acc_exit(
    variant_rows: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Plot mean cascade accuracy and exit ratio across routing thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    all_acc = []
    for key, (label, _, color) in VARIANTS.items():
        means = _mean_by_threshold(variant_rows[key])
        if not means:
            continue
        thresholds = list(means.keys())
        acc = [means[t]["cascade_acc"] * 100 for t in thresholds]
        exits = [means[t]["exit_ratio"] for t in thresholds]
        all_acc.extend(acc)
        axes[0].plot(thresholds, acc, marker="o", linewidth=1.8, label=key, color=color)
        axes[1].plot(thresholds, exits, marker="o", linewidth=1.8, label=key, color=color)

    axes[0].set_xlabel("Confidence threshold")
    axes[0].set_ylabel("Mean Cascade Accuracy (%)")
    axes[0].set_title("Accuracy vs Threshold")
    if all_acc:
        _zoom_bar_y_axis(axes[0], all_acc, lower_pad=1.0, upper_pad=1.0)
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("Mean Exit Ratio")
    axes[1].set_title("Exit Ratio vs Threshold")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)

    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Threshold Sweep Across C1-C4", y=1.02)
    fig.tight_layout()
    _save(fig, output_dir / "routing", "ablation_threshold_sweep_acc_exit")


def plot_ablation_acc_flops(variant_rows: dict[str, list[dict]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    all_flops = []
    all_acc = []
    for key, (label, _, color) in VARIANTS.items():
        rows = _fixed_rows(variant_rows[key])
        if not rows:
            continue
        flops = np.mean([r["cascade_flops"] for r in rows]) / 1e6
        acc = np.mean([r["cascade_acc"] for r in rows]) * 100
        all_flops.append(flops)
        all_acc.append(acc)
        ax.scatter(flops, acc, marker="o", s=80, color=color, label=f"{key}: {label}")
        ax.annotate(key, (flops, acc), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean Cascade FLOPs (M MACs)")
    ax.set_ylabel("Mean Cascade Accuracy (%)")
    ax.set_title("Ablation Accuracy vs FLOPs at Threshold 0.70")
    _zoom_axis_to_data(ax, all_flops, all_acc)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, output_dir / "ablations", "ablation_acc_vs_flops")


def plot_ablation_routing_calibration(variant_rows: dict[str, list[dict]], output_dir: Path) -> None:
    labels = []
    ece = []
    routing = []
    for key, rows in variant_rows.items():
        fixed_rows = _fixed_rows(rows)
        labels.append(key)
        ece.append(np.mean([r["little_ece"] for r in fixed_rows]))
        routing.append(np.mean([r["routing_error_rate"] for r in fixed_rows]))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    colors = _colors_for(labels)
    axes[0].bar(x, ece, color=colors, edgecolor="black", linewidth=0.4)
    axes[1].bar(x, routing, color=colors, edgecolor="black", linewidth=0.4)
    for ax, vals, title, ylabel in [
        (axes[0], ece, "Little-model ECE", "ECE"),
        (axes[1], routing, "Routing Error", "Routing error rate"),
    ]:
        for xpos, val in zip(x, vals):
            ax.text(xpos, val + 0.001, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{title} at Threshold 0.70")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir / "routing", "ablation_ece_routing_at_070")


def plot_independent_routing_calibration(exp_base: Path, output_dir: Path) -> None:
    independent_path = exp_base / "cifar10_independent_nas" / "combined_cascade_results.csv"
    if not independent_path.exists():
        return

    rows = sorted(
        _fixed_rows(_read_csv(independent_path)),
        key=lambda r: int(r["genotype_id"]),
    )
    if not rows:
        return

    labels = [f"Ind {int(row['genotype_id'])}" for row in rows]
    ece = [row["little_ece"] for row in rows]
    routing = [row["routing_error_rate"] for row in rows]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    colors = _colors_for(labels)
    axes[0].bar(x, ece, color=colors, edgecolor="black", linewidth=0.4)
    axes[1].bar(x, routing, color=colors, edgecolor="black", linewidth=0.4)

    for ax, vals, title, ylabel in [
        (axes[0], ece, "Little-model ECE", "ECE"),
        (axes[1], routing, "Routing Error", "Routing error rate"),
    ]:
        for xpos, val in zip(x, vals):
            ax.text(xpos, val + 0.001, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(f"Independent {title} at Threshold 0.70")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    _save(fig, output_dir / "routing", "independent_ece_routing_at_070")


def plot_ablation_ece_by_arch(variant_rows: dict[str, list[dict]], output_dir: Path) -> None:
    """Plot little-model ECE for each architecture under C1-C4."""
    arch_ids = sorted({
        row["arch_id"]
        for rows in variant_rows.values()
        for row in rows
    })
    labels = list(VARIANTS)
    x = np.arange(len(arch_ids))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset_idx, key in enumerate(labels):
        _, _, color = VARIANTS[key]
        values = []
        for arch_id in arch_ids:
            arch_rows = [
                r for r in variant_rows[key]
                if r["arch_id"] == arch_id and abs(r["threshold"] - FIXED_THRESHOLD) < 1e-9
            ]
            values.append(arch_rows[0]["little_ece"] if arch_rows else np.nan)

        positions = x + (offset_idx - (len(labels) - 1) / 2) * width
        bars = ax.bar(
            positions,
            values,
            width,
            label=f"{key}: {VARIANTS[key][0]}",
            color=color,
            edgecolor="black",
            linewidth=0.35,
        )
        ax.bar_label(bars, labels=[f"{v:.3f}" if np.isfinite(v) else "" for v in values], fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(a).replace("nas_arch_", "A") for a in arch_ids])
    ax.set_ylabel("Little-model ECE")
    ax.set_title("C1-C4 Little-model ECE by Architecture at Threshold 0.70")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    _save(fig, output_dir / "routing", "ablation_ece_by_arch_t070")


def plot_ablation_ece_training_overlays(exp_base: Path, output_dir: Path) -> None:
    """Plot validation ECE curves for C1-C4 for each retrained architecture."""
    arch_ids = sorted({
        int(path.stem.split("_")[2])
        for _, dirname, _ in VARIANTS.values()
        for path in (exp_base / dirname).glob("nas_arch_*_history.csv")
    })

    for arch_id in arch_ids:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False

        for key, (label, dirname, color) in VARIANTS.items():
            history_path = exp_base / dirname / f"nas_arch_{arch_id}_history.csv"
            if not history_path.exists():
                continue

            rows = _read_csv(history_path)
            if not rows or "val_little_ece" not in rows[0]:
                continue

            epochs = [r["epoch"] for r in rows]
            ece = [r["val_little_ece"] for r in rows]
            ax.plot(epochs, ece, color=color, linewidth=1.7, label=f"{key}: {label}")
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Little-model ECE")
        ax.set_title(f"C1-C4 ECE Training Curve Overlay - Architecture {arch_id}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        _save(fig, output_dir / "training", f"ece_training_curve_overlay_arch_{arch_id}")


def _best_by_group(rows: list[dict], group_key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return [max(items, key=lambda r: r["cascade_acc"]) for items in groups.values()]


def _best_acc_threshold_by_group(rows: list[dict], group_key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return [
        max(items, key=lambda r: (r["cascade_acc"], -r["cascade_flops"]))
        for items in groups.values()
    ]


def _candidate_key(row: dict) -> str:
    genotype = row.get("genotype")
    if genotype:
        return str(genotype)
    return "|".join(str(row.get(key, "")) for key in [
        "cascade_acc", "exit_ratio", "cascade_flops", "total_params",
        "total_bytes", "little_flops", "big_flops",
    ])


def _dedupe_candidates(rows: list[dict]) -> list[dict]:
    by_key: dict[str, dict] = {}
    for row in rows:
        key = _candidate_key(row)
        current = by_key.get(key)
        if current is None or row["cascade_acc"] > current["cascade_acc"]:
            by_key[key] = row
    return list(by_key.values())


def _select_knee_unique(rows: list[dict], k: int = 3) -> list[dict]:
    rows = sorted(_dedupe_candidates(rows), key=lambda r: r["cascade_flops"])
    n = len(rows)
    if n <= k:
        return rows

    flops = np.array([r["cascade_flops"] for r in rows], dtype=float)
    accs = np.array([r["cascade_acc"] for r in rows], dtype=float)
    f_span = float(flops.max() - flops.min())
    a_span = float(accs.max() - accs.min())
    f_n = (flops - flops.min()) / f_span if f_span > 0 else np.zeros_like(flops)
    a_n = (accs - accs.min()) / a_span if a_span > 0 else np.zeros_like(accs)

    p1 = np.array([f_n[0], a_n[0]])
    p2 = np.array([f_n[-1], a_n[-1]])
    line = p2 - p1
    line_norm = np.linalg.norm(line)
    if line_norm < 1e-12:
        i_knee = 0
    else:
        v = np.stack([f_n - p1[0], a_n - p1[1]], axis=1)
        perp = np.abs(v[:, 0] * line[1] - v[:, 1] * line[0]) / line_norm
        i_knee = int(np.argmax(perp))

    selected = [i_knee]
    offset = 1
    while len(selected) < k and (i_knee - offset >= 0 or i_knee + offset < n):
        left = i_knee - offset
        right = i_knee + offset
        if left >= 0:
            selected.append(left)
        if len(selected) >= k:
            break
        if right < n:
            selected.append(right)
        offset += 1

    return sorted([rows[i] for i in selected], key=lambda r: r["cascade_flops"])


def _non_dominated_acc_flops(rows: list[dict]) -> list[dict]:
    """Return non-dominated rows for maximize accuracy, minimize FLOPs."""
    survivors = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                other["cascade_acc"] >= row["cascade_acc"]
                and other["cascade_flops"] <= row["cascade_flops"]
                and (
                    other["cascade_acc"] > row["cascade_acc"]
                    or other["cascade_flops"] < row["cascade_flops"]
                )
            ):
                dominated = True
                break
        if not dominated:
            survivors.append(row)
    return sorted(survivors, key=lambda r: r["cascade_flops"])


def plot_proxy_unique_knee_selection(exp_base: Path, output_dir: Path) -> None:
    """Plot proxy Pareto frontier with the duplicate-safe knee selection."""
    search_dir = exp_base / "cifar10_cosearch_nas"
    search_path = search_dir / "search_log.csv"
    if not search_path.exists():
        return

    rows = [
        r for r in _read_csv(search_path)
        if isinstance(r.get("cascade_acc"), float)
        and isinstance(r.get("cascade_flops"), float)
    ]
    if not rows:
        return

    selection_pool = [
        r for r in rows
        if isinstance(r.get("little_flops"), float)
        and isinstance(r.get("big_flops"), float)
        and r["big_flops"] > r["little_flops"]
        and r["cascade_flops"] < r["big_flops"]
    ]
    if not selection_pool:
        return

    frontier = _non_dominated_acc_flops(rows)
    selection_frontier = _non_dominated_acc_flops(selection_pool)
    selected = _select_knee_unique(selection_frontier, k=3)

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.scatter(
        [r["cascade_flops"] / 1e6 for r in rows],
        [r["cascade_acc"] * 100 for r in rows],
        s=24,
        alpha=0.45,
        color="lightgray",
        label="All proxy-trained candidates",
        zorder=1,
    )
    ax.plot(
        [r["cascade_flops"] / 1e6 for r in frontier],
        [r["cascade_acc"] * 100 for r in frontier],
        color="tab:red",
        marker="o",
        linewidth=1.7,
        markersize=5,
        label="Pareto frontier over all candidates",
        zorder=3,
    )
    ax.scatter(
        [r["cascade_flops"] / 1e6 for r in selected],
        [r["cascade_acc"] * 100 for r in selected],
        marker="*",
        s=240,
        color="gold",
        edgecolor="black",
        linewidth=0.9,
        label="Knee selection with cascade FLOPs < big FLOPs",
        zorder=5,
    )
    for rank, row in enumerate(selected, start=1):
        ax.annotate(
            f"A{rank}\nexit={row['exit_ratio']:.2f}",
            xy=(row["cascade_flops"] / 1e6, row["cascade_acc"] * 100),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.85,
                "linewidth": 0.5,
            },
            zorder=6,
        )

    ax.set_xlabel("Average Cascade FLOPs (M MACs)")
    ax.set_ylabel("Proxy Cascade Accuracy (%)")
    ax.set_title("Proxy Pareto Frontier with Compute-Saving Knee Selection")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    _save(fig, output_dir / "search", "proxy_pareto_frontier_knee_unique_selected")

    print("Compute-saving knee selected architectures:")
    for rank, row in enumerate(selected, start=1):
        print(
            f"  A{rank}: acc={row['cascade_acc']:.4f}, "
            f"exit={row['exit_ratio']:.4f}, "
            f"flops={row['cascade_flops']:.0f}, "
            f"little_flops={row['little_flops']:.0f}, "
            f"big_flops={row['big_flops']:.0f}, "
            f"params={row.get('total_params', '')}"
        )


def plot_joint_independent_params(exp_base: Path, output_dir: Path) -> None:
    joint_rows = _load_variant_sweeps(exp_base)["C4"]
    joint_best = _fixed_rows(joint_rows)
    selected_path = exp_base / "cifar10_A_retrain_cascade_smooth" / "selected_architectures.csv"
    selected = _read_csv(selected_path) if selected_path.exists() else []
    params_by_arch = {
        f"nas_arch_{int(row['rank'])}": row["total_params"]
        for row in selected
    }

    independent_path = exp_base / "cifar10_independent_nas" / "combined_cascade_results.csv"
    independent_rows = _read_csv(independent_path)
    independent_best = _fixed_rows(independent_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    params_x = [params_by_arch.get(r["arch_id"], np.nan) / 1000 for r in joint_best]
    params_y = [r["cascade_acc"] * 100 for r in joint_best]
    params_x += [r["total_params"] / 1000 for r in independent_best]
    params_y += [r["cascade_acc"] * 100 for r in independent_best]
    axes[0].scatter(
        [params_by_arch.get(r["arch_id"], np.nan) / 1000 for r in joint_best],
        [r["cascade_acc"] * 100 for r in joint_best],
        s=80, color=ABLATION_COLORS["C4"], label="Joint NAS C4", edgecolor="black", linewidth=0.5,
    )
    axes[0].scatter(
        [r["total_params"] / 1000 for r in independent_best],
        [r["cascade_acc"] * 100 for r in independent_best],
        s=55, color=INDEPENDENT_COLOR, label="Independent NAS", marker="s", edgecolor="black", linewidth=0.4,
    )
    axes[0].set_xlabel("Combined Params (K)")
    axes[0].set_ylabel("Cascade Accuracy (%)")
    axes[0].set_title("Accuracy vs Model Size at Threshold 0.70")
    _zoom_axis_to_data(
        axes[0],
        [x for x in params_x if np.isfinite(x)],
        [y for x, y in zip(params_x, params_y) if np.isfinite(x)],
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    flops_x = [r["cascade_flops"] / 1e6 for r in joint_best + independent_best]
    flops_y = [r["cascade_acc"] * 100 for r in joint_best + independent_best]
    axes[1].scatter(
        [r["cascade_flops"] / 1e6 for r in joint_best],
        [r["cascade_acc"] * 100 for r in joint_best],
        s=80, color=ABLATION_COLORS["C4"], label="Joint NAS C4", edgecolor="black", linewidth=0.5,
    )
    axes[1].scatter(
        [r["cascade_flops"] / 1e6 for r in independent_best],
        [r["cascade_acc"] * 100 for r in independent_best],
        s=55, color=INDEPENDENT_COLOR, label="Independent NAS", marker="s", edgecolor="black", linewidth=0.4,
    )
    axes[1].set_xlabel("Cascade FLOPs at Threshold 0.70 (M MACs)")
    axes[1].set_ylabel("Cascade Accuracy (%)")
    axes[1].set_title("Accuracy vs Compute at Threshold 0.70")
    _zoom_axis_to_data(axes[1], flops_x, flops_y)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    _save(fig, output_dir / "rq1", "joint_vs_independent_size_compute")


def plot_c1_vs_independent_acc_flops(exp_base: Path, output_dir: Path) -> None:
    """Plot fully retrained C1 co-search cascades against independent cascades."""
    joint_rows = _fixed_rows(_load_variant_sweeps(exp_base)["C1"])
    independent_path = exp_base / "cifar10_independent_nas" / "combined_cascade_results.csv"
    if not joint_rows or not independent_path.exists():
        return

    independent_rows = _fixed_rows(_read_csv(independent_path))
    if not independent_rows:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    all_flops = [r["cascade_flops"] / 1e6 for r in joint_rows + independent_rows]
    all_acc = [r["cascade_acc"] * 100 for r in joint_rows + independent_rows]

    ax.scatter(
        [r["cascade_flops"] / 1e6 for r in joint_rows],
        [r["cascade_acc"] * 100 for r in joint_rows],
        s=85,
        color=ABLATION_COLORS["C1"],
        label="Co-search C1 (CE only)",
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    ax.scatter(
        [r["cascade_flops"] / 1e6 for r in independent_rows],
        [r["cascade_acc"] * 100 for r in independent_rows],
        s=55,
        color=INDEPENDENT_COLOR,
        marker="s",
        label="Independent NAS",
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )

    for row in joint_rows:
        label = str(row["arch_id"]).replace("nas_arch_", "A")
        ax.annotate(
            label,
            xy=(row["cascade_flops"] / 1e6, row["cascade_acc"] * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("Average Cascade FLOPs at Threshold 0.70 (M MACs)")
    ax.set_ylabel("Cascade Accuracy (%)")
    ax.set_title("Fully Trained C1 Co-search vs Independent NAS")
    _zoom_axis_to_data(ax, all_flops, all_acc)
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output_dir / "rq1", "rq1_c1_vs_independent_t070")


def plot_c1_vs_independent_best_threshold(exp_base: Path, output_dir: Path) -> None:
    """Plot C1 at threshold 0.70 against independent cascades at best threshold."""
    joint_rows = _fixed_rows(_load_variant_sweeps(exp_base)["C1"])
    independent_path = exp_base / "cifar10_independent_nas" / "combined_cascade_results.csv"
    if not joint_rows or not independent_path.exists():
        return

    independent_rows = _best_acc_threshold_by_group(
        _read_csv(independent_path),
        "genotype_id",
    )
    independent_rows = sorted(independent_rows, key=lambda r: int(r["genotype_id"]))
    if not independent_rows:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    all_flops = [r["cascade_flops"] / 1e6 for r in joint_rows + independent_rows]
    all_acc = [r["cascade_acc"] * 100 for r in joint_rows + independent_rows]

    ax.scatter(
        [r["cascade_flops"] / 1e6 for r in joint_rows],
        [r["cascade_acc"] * 100 for r in joint_rows],
        s=85,
        color=ABLATION_COLORS["C1"],
        label="Co-search C1 at threshold 0.70",
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    ax.scatter(
        [r["cascade_flops"] / 1e6 for r in independent_rows],
        [r["cascade_acc"] * 100 for r in independent_rows],
        s=60,
        color=INDEPENDENT_COLOR,
        marker="s",
        label="Independent NAS at best threshold",
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )

    for row in joint_rows:
        label = str(row["arch_id"]).replace("nas_arch_", "A")
        ax.annotate(
            label,
            xy=(row["cascade_flops"] / 1e6, row["cascade_acc"] * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    for row in independent_rows:
        ax.annotate(
            f"I{int(row['genotype_id'])}\nt={row['threshold']:.2f}",
            xy=(row["cascade_flops"] / 1e6, row["cascade_acc"] * 100),
            xytext=(5, -14),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("Average Cascade FLOPs (M MACs)")
    ax.set_ylabel("Cascade Accuracy (%)")
    ax.set_title("C1 Co-search vs Independent NAS with Best Independent Threshold")
    _zoom_axis_to_data(ax, all_flops, all_acc)
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output_dir / "rq1", "rq1_c1_t070_vs_independent_best_threshold")


def plot_exit_accuracy(exp_base: Path, output_dir: Path) -> None:
    joint_rows = _fixed_rows(_load_variant_sweeps(exp_base)["C4"])
    independent_rows = _fixed_rows(_read_csv(exp_base / "cifar10_independent_nas" / "combined_cascade_results.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))
    all_exit = [r["exit_ratio"] for r in independent_rows + joint_rows]
    all_acc = [r["cascade_acc"] * 100 for r in independent_rows + joint_rows]
    ax.scatter(
        [r["exit_ratio"] for r in independent_rows],
        [r["cascade_acc"] * 100 for r in independent_rows],
        s=28, alpha=0.55, color=INDEPENDENT_COLOR, label="Independent NAS",
    )
    ax.scatter(
        [r["exit_ratio"] for r in joint_rows],
        [r["cascade_acc"] * 100 for r in joint_rows],
        s=65, color=ABLATION_COLORS["C4"], label="Joint NAS C4", edgecolor="black", linewidth=0.4,
    )
    ax.set_xlabel("Exit Ratio")
    ax.set_ylabel("Cascade Accuracy (%)")
    ax.set_title("Accuracy vs Early-exit Ratio at Threshold 0.70")
    _zoom_axis_to_data(ax, all_exit, all_acc)
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output_dir / "routing", "accuracy_vs_exit_ratio")


def _plot_flops_breakdown(
    rows: list[dict],
    labels: list[str],
    title: str,
    output_dir: Path,
    name: str,
    zoom_y: bool = False,
) -> None:
    x = np.arange(len(rows))
    width = 0.25
    little = [r["little_flops"] / 1e6 for r in rows]
    big = [r["big_flops"] / 1e6 for r in rows]
    cascade = [r["cascade_flops"] / 1e6 for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.2), 5))
    little_bars = ax.bar(x - width, little, width, label="Little FLOPs", color="#4C78A8")
    big_bars = ax.bar(x, big, width, label="Big FLOPs", color="#F58518")
    cascade_bars = ax.bar(x + width, cascade, width, label="Cascade FLOPs", color="#54A24B")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("FLOPs (M MACs)")
    ax.set_title(title)
    if zoom_y:
        values = little + big + cascade
        y_min = min(values)
        y_max = max(values)
        y_span = max(y_max - y_min, max(abs(y_max), 1.0) * 0.08)
        ax.set_ylim(max(0.0, y_min - y_span * 0.12), y_max + y_span * 0.28)
    else:
        ax.set_ylim(0, max(little + big + cascade) * 1.18)

    for bars, values in ((little_bars, little), (big_bars, big)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}M",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    for bar, cascade_value, big_value in zip(cascade_bars, cascade, big):
        pct_of_big = cascade_value / big_value * 100 if big_value else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{cascade_value:.2f}M\n{pct_of_big:.0f}% of big",
            ha="center",
            va="bottom",
            fontsize=7,
            linespacing=0.9,
        )

    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, output_dir, name)


def plot_flops_breakdowns(exp_base: Path, output_dir: Path) -> None:
    """Plot little, big, and expected cascade FLOPs.

    Joint uses the C3 cascade-aware-loss sweeps at threshold 0.70. Independent
    uses the five most accurate genotypes at threshold 0.70 to keep the chart
    readable while still showing the scale difference.
    """
    joint_rows = []
    joint_labels = []
    joint_dir = exp_base / "cifar10_A_retrain_cascade_only"
    for idx, path in enumerate(sorted(joint_dir.glob("nas_arch_*_threshold_sweep.csv")), start=1):
        if "tempscaled" in path.name:
            continue
        rows = _read_csv(path)
        row = next((r for r in rows if abs(r["threshold"] - FIXED_THRESHOLD) < 1e-9), None)
        if row is not None:
            joint_rows.append(row)
            joint_labels.append(f"Joint A{idx}")

    if joint_rows:
        _plot_flops_breakdown(
            joint_rows,
            joint_labels,
            "Joint C3 FLOPs Breakdown at Threshold 0.70",
            output_dir / "flops",
            "flops_breakdown_joint_c3_t070",
        )

    independent_path = exp_base / "cifar10_independent_nas" / "combined_cascade_results.csv"
    if independent_path.exists():
        independent_rows = [
            r for r in _read_csv(independent_path)
            if abs(r["threshold"] - FIXED_THRESHOLD) < 1e-9
        ]
        independent_rows = sorted(
            independent_rows,
            key=lambda r: r["cascade_acc"],
            reverse=True,
        )[:5]
        if independent_rows:
            _plot_flops_breakdown(
                independent_rows,
                [f"Ind {int(r['genotype_id'])}" for r in independent_rows],
                "Top Independent FLOPs Breakdown at Threshold 0.70",
                output_dir / "flops",
                "flops_breakdown_independent_top5_t070",
            )


def plot_ablation_flops_breakdown(variant_rows: dict[str, list[dict]], output_dir: Path) -> None:
    """Plot mean little, big, and cascade FLOPs for C1-C4 at threshold 0.70."""
    rows = []
    labels = []
    for key, variant_data in variant_rows.items():
        fixed_rows = _fixed_rows(variant_data)
        if not fixed_rows:
            continue
        rows.append({
            "little_flops": float(np.mean([r["little_flops"] for r in fixed_rows])),
            "big_flops": float(np.mean([r["big_flops"] for r in fixed_rows])),
            "cascade_flops": float(np.mean([r["cascade_flops"] for r in fixed_rows])),
        })
        labels.append(key)

    if rows:
        _plot_flops_breakdown(
            rows,
            labels,
            "C1-C4 Mean FLOPs Breakdown at Threshold 0.70",
            output_dir / "flops",
            "flops_breakdown_ablation_c1_c4_t070",
            zoom_y=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate extra result plots from saved CSVs.")
    parser.add_argument("--experiments-root", default="experiments", type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args()

    exp_base = args.experiments_root
    output_dir = args.output_dir or exp_base / "plots"
    variant_rows = _load_variant_sweeps(exp_base)

    plot_ablation_training_overlays(exp_base, output_dir)
    plot_ablation_ece_training_overlays(exp_base, output_dir)
    plot_proxy_unique_knee_selection(exp_base, output_dir)
    plot_ablation_accuracy_bars(variant_rows, output_dir)
    plot_ablation_accuracy_bars_by_arch(variant_rows, output_dir)
    plot_ablation_little_accuracy_bars(variant_rows, output_dir)
    plot_ablation_little_accuracy_bars_by_arch(variant_rows, output_dir)
    plot_ablation_big_accuracy_bars(variant_rows, output_dir)
    plot_ablation_big_accuracy_bars_by_arch(variant_rows, output_dir)
    plot_ablation_threshold_sensitivity(variant_rows, output_dir)
    plot_ablation_threshold_sweep_acc_exit(variant_rows, output_dir)
    plot_ablation_acc_flops(variant_rows, output_dir)
    plot_ablation_routing_calibration(variant_rows, output_dir)
    plot_independent_routing_calibration(exp_base, output_dir)
    plot_ablation_ece_by_arch(variant_rows, output_dir)
    plot_joint_independent_params(exp_base, output_dir)
    plot_c1_vs_independent_acc_flops(exp_base, output_dir)
    plot_c1_vs_independent_best_threshold(exp_base, output_dir)
    plot_exit_accuracy(exp_base, output_dir)
    plot_flops_breakdowns(exp_base, output_dir)
    plot_ablation_flops_breakdown(variant_rows, output_dir)


if __name__ == "__main__":
    main()
