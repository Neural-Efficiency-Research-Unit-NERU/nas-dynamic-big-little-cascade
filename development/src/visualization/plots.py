"""Publication-quality plots for edge NAS experiments."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.training.utils import compute_ece


STYLE = "seaborn-v0_8-paper"
DPI = 300


def _setup_style():
    plt.style.use(STYLE)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": DPI,
    })


def _save(fig, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(path / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.png / .pdf")


def load_training_history(csv_path: Path) -> dict:
    data = {"epoch": [], "train_loss": [], "val_loss": []}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["train_loss"].append(float(row["train_loss"]))
            data["val_loss"].append(float(row.get("val_loss", 0)))
            if "val_cascade_acc" in row:
                data.setdefault("val_cascade_acc", []).append(float(row["val_cascade_acc"]))
                data.setdefault("val_little_acc", []).append(float(row["val_little_acc"]))
                data.setdefault("val_big_acc", []).append(float(row["val_big_acc"]))
            elif "val_acc" in row:
                data.setdefault("val_acc", []).append(float(row["val_acc"]))
            if "train_acc" in row:
                data.setdefault("train_acc", []).append(float(row["train_acc"]))
    return data


def plot_training_curves(history: dict, model_name: str, output_dir: Path):
    """Training curves: loss + accuracy vs epoch."""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    epochs = history["epoch"]
    ax1.plot(epochs, history["train_loss"], label="Train", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], label="Validation", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if "val_cascade_acc" in history:
        ax2.plot(epochs, [a * 100 for a in history["val_cascade_acc"]], label="Cascade", linewidth=1.5)
        ax2.plot(epochs, [a * 100 for a in history["val_little_acc"]], label="Little", linewidth=1.5, linestyle="--")
        ax2.plot(epochs, [a * 100 for a in history["val_big_acc"]], label="Big", linewidth=1.5, linestyle=":")
    elif "val_acc" in history:
        ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation", linewidth=1.5)
        if "train_acc" in history:
            ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{model_name} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} Training Curves", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir, f"training_curves_{model_name.lower().replace(' ', '_')}")


def plot_nas_search_progress(search_log_path: Path, output_dir: Path):
    """Scatter plot of NAS evaluations vs generation. Primary axis is avg cascade FLOPs.

    NAS optimizes cascade accuracy and average cascade FLOPs, so the
    search-progress visualization follows those two quantities.
    """
    _setup_style()
    data = {"gen": [], "acc": [], "flops": []}
    with open(search_log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("cascade_flops"):
                continue
            data["gen"].append(int(row["generation"]))
            data["acc"].append(float(row["cascade_acc"]) * 100)
            data["flops"].append(float(row["cascade_flops"]) / 1e6)

    if not data["acc"]:
        print("  Warning: nas_search_progress: search_log.csv has no cascade_flops "
              "rows. Re-run with the current evolutionary search implementation.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        data["flops"], data["acc"],
        c=data["gen"], cmap="viridis", s=40, alpha=0.7, edgecolors="black", linewidth=0.3,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Generation")
    ax.set_xlabel("Average Cascade FLOPs (M MACs)")
    ax.set_ylabel("Proxy Cascade Accuracy (%)")
    ax.set_title("NAS Search Progress (memory hard constraint = 450 KB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "nas_search_progress")


def plot_pareto_frontier(
    search_log_path: Path,
    pareto_path: Path,
    output_dir: Path,
    selected_path: Path | None = None,
):
    """Joint-NAS Pareto frontier: cascade accuracy vs avg cascade FLOPs.

    The X-axis is the actual NAS objective, average cascade FLOPs, rather than
    parameter count.
    """
    _setup_style()

    all_acc, all_flops = [], []
    with open(search_log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("cascade_flops"):
                continue
            all_acc.append(float(row["cascade_acc"]) * 100)
            all_flops.append(float(row["cascade_flops"]) / 1e6)

    pareto_acc, pareto_flops = [], []
    with open(pareto_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("cascade_flops"):
                continue
            pareto_acc.append(float(row["cascade_acc"]) * 100)
            pareto_flops.append(float(row["cascade_flops"]) / 1e6)

    if not pareto_flops:
        print("  Warning: pareto_frontier: pareto_front.csv has no cascade_flops "
              "rows. Re-run with the current evolutionary search implementation.")
        return

    sorted_idx = np.argsort(pareto_flops)
    pareto_flops_sorted = [pareto_flops[i] for i in sorted_idx]
    pareto_acc_sorted = [pareto_acc[i] for i in sorted_idx]

    selected = []
    if selected_path is not None and selected_path.exists():
        with open(selected_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("cascade_flops"):
                    continue
                selected.append({
                    "rank": row.get("rank", ""),
                    "acc": float(row["cascade_acc"]) * 100,
                    "flops": float(row["cascade_flops"]) / 1e6,
                    "exit_ratio": row.get("exit_ratio", ""),
                })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(all_flops, all_acc, c="lightgray", s=20, alpha=0.5,
               label="NAS candidates", zorder=1)
    ax.plot(pareto_flops_sorted, pareto_acc_sorted, "r-o", markersize=6, linewidth=1.5,
            label="Pareto front (joint NAS)", zorder=3)
    if selected:
        sel_flops = [r["flops"] for r in selected]
        sel_acc = [r["acc"] for r in selected]
        ax.scatter(
            sel_flops,
            sel_acc,
            marker="*",
            s=190,
            c="gold",
            edgecolors="black",
            linewidths=0.9,
            label="Selected for retrain",
            zorder=5,
        )
        for row in selected:
            label = f"A{row['rank']}"
            if row["exit_ratio"]:
                label += f"\nexit={float(row['exit_ratio']):.2f}"
            ax.annotate(
                label,
                xy=(row["flops"], row["acc"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "alpha": 0.8,
                    "linewidth": 0.5,
                },
                zorder=6,
            )

    ax.set_xlabel("Average Cascade FLOPs (M MACs)")
    ax.set_ylabel("Cascade Accuracy (%)")
    ax.set_title("Pareto Frontier: Accuracy vs. Avg Cascade FLOPs")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "pareto_frontier")


def plot_reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
    model_name: str = "Little Model",
    output_dir: Path | None = None,
):
    """Reliability diagram: observed accuracy vs predicted confidence per bin."""
    _setup_style()

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_width = 1.0 / n_bins

    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        count = mask.sum()
        if count > 0:
            bin_accs[i] = accuracies[mask].mean()
            bin_confs[i] = confidences[mask].mean()
            bin_counts[i] = count

    ece = compute_ece(confidences, accuracies, n_bins)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    # Bar chart of observed accuracy per bin
    non_empty = bin_counts > 0
    ax.bar(
        bin_centers[non_empty], bin_accs[non_empty],
        width=bin_width * 0.9, alpha=0.7, color="steelblue",
        edgecolor="black", linewidth=0.5, label="Observed accuracy",
    )

    # Shade gap areas (miscalibration)
    for i in range(n_bins):
        if bin_counts[i] == 0:
            continue
        lo_val = min(bin_accs[i], bin_confs[i])
        hi_val = max(bin_accs[i], bin_confs[i])
        ax.bar(
            bin_centers[i], hi_val - lo_val, bottom=lo_val,
            width=bin_width * 0.9, alpha=0.3, color="red",
            edgecolor="none",
        )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram - {model_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.05, 0.92, f"ECE = {ece:.4f}",
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_dir is not None:
        _save(fig, output_dir, f"reliability_{model_name.lower().replace(' ', '_')}")
    else:
        plt.close(fig)

    return fig






def _zoom_axis_to_data(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    x_pad_fraction: float = 0.08,
    y_pad_fraction: float = 0.12,
) -> None:
    """Set plot limits around the data instead of anchoring axes at zero."""
    if not xs or not ys:
        return

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, max(abs(x_max), 1.0) * 0.05)
    y_span = max(y_max - y_min, max(abs(y_max), 1.0) * 0.02)

    ax.set_xlim(x_min - x_span * x_pad_fraction, x_max + x_span * x_pad_fraction)
    ax.set_ylim(y_min - y_span * y_pad_fraction, y_max + y_span * y_pad_fraction)


def plot_acc_vs_flops_curve(
    deferral_data: list[dict],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Cascade accuracy vs avg cascade FLOPs as threshold varies.

    Sibling of `plot_deferral_curve` (which uses exit_ratio on the X-axis).
    This is the cleaner reading of the same data: at a given compute budget
    (avg FLOPs), what accuracy do we get?
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for d in deferral_data:
        flops_m = [f / 1e6 for f in d["cascade_flops"]]
        accs = [a * 100 for a in d["cascade_accs"]]
        thresholds = d["thresholds"]
        ax.plot(flops_m, accs, color=d["color"], linewidth=1.5,
                marker="o", markersize=5, label=d["name"])
        for t, fl, ca in zip(thresholds, flops_m, accs):
            ax.annotate(f"{t:.2f}", (fl, ca),
                        textcoords="offset points", xytext=(4, 5),
                        fontsize=7, color=d["color"])

    ax.set_xlabel("Average Cascade FLOPs (M MACs)")
    ax.set_ylabel("Cascade Accuracy (%)")
    ax.set_title("Cascade Accuracy vs. FLOPs (threshold-swept)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_dir is not None:
        _save(fig, output_dir, "acc_vs_flops_curve")
    else:
        plt.close(fig)

    return fig


def plot_ablation_comparison(
    results: list[dict],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing C1/C2/C3 ablation results.

    Args:
        results: list of {name: str, cascade_acc: float, exit_ratio: float,
                         little_ece: float, little_ece_post: float}
    """
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    names = [r["name"] for r in results]
    colors = {"C1": "#1f77b4", "C2": "#ff7f0e", "C3": "#9467bd", "C4": "#2ca02c"}
    bar_colors = [colors.get(n, plt.cm.Set2(i / len(names))) for i, n in enumerate(names)]
    x = np.arange(len(names))
    bar_width = 0.5

    # Top-left: Cascade Accuracy
    ax = axes[0, 0]
    vals = [r["cascade_acc"] * 100 for r in results]
    bars = ax.bar(x, vals, width=bar_width, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Cascade Accuracy (%)")
    ax.set_title("Cascade Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.grid(True, alpha=0.3, axis="y")

    # Top-right: Exit Ratio
    ax = axes[0, 1]
    vals = [r["exit_ratio"] for r in results]
    bars = ax.bar(x, vals, width=bar_width, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Exit Ratio")
    ax.set_title("Exit Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left: Little ECE (pre-temp)
    ax = axes[1, 0]
    vals = [r["little_ece"] for r in results]
    bars = ax.bar(x, vals, width=bar_width, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ECE")
    ax.set_title("Little ECE (pre-temp)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: Little ECE (post-temp)
    ax = axes[1, 1]
    vals = [r["little_ece_post"] for r in results]
    bars = ax.bar(x, vals, width=bar_width, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ECE")
    ax.set_title("Little ECE (post-temp)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("RQ2: Training Strategy Ablation", fontsize=13, y=1.02)
    fig.tight_layout()

    if output_dir is not None:
        _save(fig, output_dir, "ablation_comparison")
    else:
        plt.close(fig)

    return fig


def plot_deferral_curve(
    deferral_data: list[dict],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Deferral curve: cascade accuracy vs exit ratio across thresholds.

    Args:
        deferral_data: list of {name: str, thresholds: list, cascade_accs: list,
                       exit_ratios: list, color: str}
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for d in deferral_data:
        exit_ratios = d["exit_ratios"]
        cascade_accs = d["cascade_accs"]
        thresholds = d["thresholds"]
        ax.plot(
            exit_ratios, cascade_accs,
            color=d["color"], linewidth=1.5, label=d["name"],
        )
        ax.scatter(exit_ratios, cascade_accs, color=d["color"], s=30, zorder=3)
        # Annotate threshold values at each point
        for t, er, ca in zip(thresholds, exit_ratios, cascade_accs):
            ax.annotate(
                f"{t:.2f}", (er, ca),
                textcoords="offset points", xytext=(4, 6),
                fontsize=7, color=d["color"],
            )

    ax.set_xlabel("Exit Ratio")
    ax.set_ylabel("Cascade Accuracy (%)")
    ax.set_xlim(0, 1)
    ax.set_title("Deferral Curve: Accuracy vs. Exit Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_dir is not None:
        _save(fig, output_dir, "deferral_curve")
    else:
        plt.close(fig)

    return fig


def plot_ece_comparison(
    ece_data: list[dict],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Bar chart comparing ECE before and after temperature scaling.

    Args:
        ece_data: list of {name: str, ece_pre: float, ece_post: float,
                  learned_temp: float}
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    names = [d["name"] for d in ece_data]
    ece_pre = [d["ece_pre"] for d in ece_data]
    ece_post = [d["ece_post"] for d in ece_data]
    temps = [d["learned_temp"] for d in ece_data]

    colors = {"C1": "#1f77b4", "C2": "#ff7f0e", "C3": "#9467bd", "C4": "#2ca02c"}
    bar_colors = [colors.get(n, plt.cm.Set2(i / len(names))) for i, n in enumerate(names)]

    x = np.arange(len(names))
    bar_width = 0.3

    bars_pre = ax.bar(
        x - bar_width / 2, ece_pre, width=bar_width,
        color=bar_colors, edgecolor="black", linewidth=0.5,
        hatch="//", alpha=0.7, label="Pre-temp scaling",
    )
    bars_post = ax.bar(
        x + bar_width / 2, ece_post, width=bar_width,
        color=bar_colors, edgecolor="black", linewidth=0.5,
        alpha=0.9, label="Post-temp scaling",
    )

    for i, (bar_pre, bar_post, t) in enumerate(zip(bars_pre, bars_post, temps)):
        ax.text(
            bar_pre.get_x() + bar_pre.get_width() / 2, bar_pre.get_height() + 0.001,
            f"{ece_pre[i]:.4f}", ha="center", va="bottom", fontsize=8,
        )
        ax.text(
            bar_post.get_x() + bar_post.get_width() / 2, bar_post.get_height() + 0.001,
            f"{ece_post[i]:.4f}", ha="center", va="bottom", fontsize=8,
        )
        # Annotate learned temperature below the group
        ax.text(
            x[i], -0.008, f"T={t:.2f}",
            ha="center", va="top", fontsize=8, fontstyle="italic",
        )

    ax.set_ylabel("ECE")
    ax.set_title("Calibration: ECE Before/After Temperature Scaling")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if output_dir is not None:
        _save(fig, output_dir, "ece_comparison")
    else:
        plt.close(fig)

    return fig


def plot_training_curve_overlay(
    histories: list[dict],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Overlay validation cascade accuracy curves from multiple experiments.

    Args:
        histories: list of {name: str, epochs: list, val_cascade_acc: list,
                   color: str}
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for h in histories:
        accs = [a * 100 for a in h["val_cascade_acc"]]
        ax.plot(
            h["epochs"], accs,
            color=h["color"], linewidth=1.5, label=h["name"],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Cascade Accuracy (%)")
    ax.set_title("Training Dynamics: Cascade Accuracy Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_dir is not None:
        _save(fig, output_dir, "training_curve_overlay")
    else:
        plt.close(fig)

    return fig


def plot_threshold_sensitivity(
    threshold_data: list[dict],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Show cascade accuracy and exit ratio vs threshold for multiple variants.

    Args:
        threshold_data: list of {name: str, thresholds: list, cascade_accs: list,
                        exit_ratios: list, color: str}
    """
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for d in threshold_data:
        ax1.plot(
            d["thresholds"], d["cascade_accs"],
            color=d["color"], linewidth=1.5, label=d["name"],
        )
        ax2.plot(
            d["thresholds"], d["exit_ratios"],
            color=d["color"], linewidth=1.5, label=d["name"],
        )

    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Cascade Accuracy (%)")
    ax1.set_title("Cascade Accuracy vs Threshold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Exit Ratio")
    ax2.set_title("Exit Ratio vs Threshold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Threshold Sensitivity Analysis", fontsize=13, y=1.02)
    fig.tight_layout()

    if output_dir is not None:
        _save(fig, output_dir, "threshold_sensitivity")
    else:
        plt.close(fig)

    return fig
