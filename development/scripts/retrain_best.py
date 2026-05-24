#!/usr/bin/env python
"""Retrain selected joint NAS architectures and write threshold-sweep results."""
import ast
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.data import get_data_loaders
from src.models.paired_models import CascadePair
from src.nas.search_space import PairGenotype
from src.training.trainer import train_joint_model
from src.training.utils import (
    cascade_flops,
    count_parameters,
    estimate_pair_flops,
    estimate_pair_memory,
    MEMORY_BUDGET_BYTES,
    PARAM_BUDGET_REFERENCE,
    evaluate_at_thresholds,
    get_device,
    get_experiment_dir,
    learn_temperature,
    load_config,
    set_seed,
)


DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def _candidate_key(row: dict) -> str:
    genotype = row.get("genotype")
    if genotype is not None:
        return str(genotype.to_dict())
    return str(row)


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    """Keep the best proxy-accuracy row for each genotype."""
    by_key: dict[str, dict] = {}
    for row in candidates:
        key = _candidate_key(row)
        current = by_key.get(key)
        if current is None or row["cascade_acc"] > current["cascade_acc"]:
            by_key[key] = row
    return list(by_key.values())


def filter_budget_feasible(
    candidates: list[dict],
    min_params: int | None = None,
    max_params: int | None = PARAM_BUDGET_REFERENCE,
    max_bytes: int | None = MEMORY_BUDGET_BYTES,
    require_big_flops_gt_little: bool = False,
    require_cascade_flops_lt_big: bool = False,
) -> list[dict]:
    """Filter candidates by the deployment budgets used for final retraining."""
    filtered = []
    for row in candidates:
        min_params_ok = min_params is None or (
            row.get("total_params") is not None and row["total_params"] >= min_params
        )
        params_ok = max_params is None or (
            row.get("total_params") is not None and row["total_params"] <= max_params
        )
        bytes_ok = max_bytes is None or (
            row.get("total_bytes") is not None and row["total_bytes"] <= max_bytes
        )
        flops_ok = _finite(row.get("cascade_flops"))
        role_flops_ok = True
        if require_big_flops_gt_little:
            role_flops_ok = (
                _finite(row.get("little_flops"))
                and _finite(row.get("big_flops"))
                and row["big_flops"] > row["little_flops"]
            )
        cascade_saves_flops_ok = True
        if require_cascade_flops_lt_big:
            cascade_saves_flops_ok = (
                _finite(row.get("cascade_flops"))
                and _finite(row.get("big_flops"))
                and row["cascade_flops"] < row["big_flops"]
            )
        if (
            min_params_ok
            and params_ok
            and bytes_ok
            and flops_ok
            and role_flops_ok
            and cascade_saves_flops_ok
        ):
            filtered.append(row)
    return filtered


def _fill_unique(
    picks: list[dict],
    pool: list[dict],
    k: int,
) -> list[dict]:
    seen = {_candidate_key(p) for p in picks}
    for row in pool:
        key = _candidate_key(row)
        if key not in seen:
            picks.append(row)
            seen.add(key)
        if len(picks) >= k:
            break
    return picks


def select_balanced_candidates(
    candidates: list[dict],
    k: int = 3,
    exit_low: float = 0.2,
    exit_high: float = 0.8,
    acc_tolerance: float = 0.02,
) -> tuple[list[dict], str]:
    """Select final retrain candidates with an exit-ratio guard.

    If the search produced real cascades, choose a diverse set around the useful
    routing region: best accuracy, cheapest near-best, and highest-exit near-best.
    If every candidate is degenerate, fall back to the best proxy-accuracy
    candidates instead of retraining low-FLOP knee points that are also
    degenerate.
    """
    if not candidates:
        raise ValueError("select_balanced_candidates: no candidates after budget filtering")

    candidates = _dedupe_candidates(candidates)
    balanced = [
        row for row in candidates
        if row.get("exit_ratio") is not None and exit_low <= row["exit_ratio"] <= exit_high
    ]

    if not balanced:
        return (
            sorted(candidates, key=lambda r: r["cascade_acc"], reverse=True)[:k],
            "no candidates in exit-ratio window; fell back to top proxy accuracy",
        )

    best_acc = max(row["cascade_acc"] for row in balanced)
    near_best = [
        row for row in balanced
        if row["cascade_acc"] >= best_acc - acc_tolerance
    ]

    picks: list[dict] = []
    _fill_unique(
        picks,
        [
            max(balanced, key=lambda r: r["cascade_acc"]),
            min(near_best, key=lambda r: r["cascade_flops"]),
            max(near_best, key=lambda r: r["exit_ratio"]),
        ],
        k,
    )
    _fill_unique(
        picks,
        sorted(balanced, key=lambda r: (-r["cascade_acc"], r["cascade_flops"])),
        k,
    )
    _fill_unique(
        picks,
        sorted(candidates, key=lambda r: (-r["cascade_acc"], r["cascade_flops"])),
        k,
    )
    return picks[:k], "selected balanced exit-ratio candidates"


def select_knee_neighbors(pareto: list[dict], k: int = 3) -> list[dict]:
    """Pick the knee point of the (cascade_flops, cascade_acc) front plus
    k - 1 nearest-by-FLOPs neighbours.

    Knee = Pareto point with maximum perpendicular distance to the line
    connecting the FLOPs-minimum and accuracy-maximum anchors of the
    normalised front.

    Edge cases:
    - Empty front: raises ValueError.
    - Front with fewer than k points: returns all available, sorted by FLOPs.
    - Knee at an endpoint: window slides toward the interior.
    """
    if not pareto:
        raise ValueError("select_knee_neighbors: empty Pareto front")

    pareto = _dedupe_candidates(pareto)
    pts = sorted(pareto, key=lambda r: r["cascade_flops"])
    n = len(pts)
    if n <= k:
        return pts

    flops = np.array([r["cascade_flops"] for r in pts], dtype=float)
    accs = np.array([r["cascade_acc"] for r in pts], dtype=float)

    f_span = float(flops.max() - flops.min())
    a_span = float(accs.max() - accs.min())
    f_n = (flops - flops.min()) / f_span if f_span > 0 else np.zeros_like(flops)
    a_n = (accs - accs.min()) / a_span if a_span > 0 else np.zeros_like(accs)

    p1 = np.array([f_n[0], a_n[0]])
    p2 = np.array([f_n[-1], a_n[-1]])
    line = p2 - p1
    line_norm = np.linalg.norm(line)
    if line_norm < 1e-12:
        # Degenerate: all points coincide. Treat knee as leftmost.
        i_knee = 0
    else:
        # Use 2-D scalar cross product (np.cross on 2-D vectors is deprecated in numpy 2.0).
        v = np.stack([f_n - p1[0], a_n - p1[1]], axis=1)
        perp = np.abs(v[:, 0] * line[1] - v[:, 1] * line[0]) / line_norm
        i_knee = int(np.argmax(perp))

    selected_indices = [i_knee]
    offset = 1
    while len(selected_indices) < k and (i_knee - offset >= 0 or i_knee + offset < n):
        left = i_knee - offset
        right = i_knee + offset
        if left >= 0:
            selected_indices.append(left)
        if len(selected_indices) >= k:
            break
        if right < n:
            selected_indices.append(right)
        offset += 1

    picks = [pts[i] for i in selected_indices]
    return sorted(picks, key=lambda r: r["cascade_flops"])


def non_dominated_acc_flops(candidates: list[dict]) -> list[dict]:
    """Return non-dominated candidates for maximize accuracy, minimize FLOPs."""
    candidates = _dedupe_candidates(candidates)
    survivors = []
    for row in candidates:
        dominated = False
        for other in candidates:
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


def load_pareto_front(search_dir: Path) -> list[dict]:
    """Load Pareto front from NAS search results.

    Tolerates the older schema (no `cascade_flops`/`total_bytes` columns) so a
    rerun against a stale `pareto_front.csv` doesn't fail. Derives `exit_ratio`
    analytically from `(cascade_flops, little_flops, big_flops)` when all three
    are present:

        cascade_flops = little_flops + (1 - exit_ratio) * big_flops
        => exit_ratio = 1 - (cascade_flops - little_flops) / big_flops

    Returns rows in CSV order; selection logic decides ordering.
    """
    pareto_path = search_dir / "pareto_front.csv"
    results = []
    with open(pareto_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cflops = float(row["cascade_flops"]) if row.get("cascade_flops") else None
            lflops = float(row["little_flops"]) if row.get("little_flops") else None
            bflops = float(row["big_flops"]) if row.get("big_flops") else None
            exit_ratio = float(row["exit_ratio"]) if row.get("exit_ratio") else None
            if exit_ratio is None and cflops is not None and lflops is not None and bflops and bflops > 0:
                exit_ratio = 1.0 - (cflops - lflops) / bflops
            results.append({
                "cascade_acc": float(row["cascade_acc"]),
                "total_params": int(row["total_params"]),
                "cascade_flops": cflops,
                "little_flops": lflops,
                "big_flops": bflops,
                "exit_ratio": exit_ratio,
                "total_bytes": int(row["total_bytes"]) if row.get("total_bytes") else None,
                "genotype": PairGenotype.from_dict(ast.literal_eval(row["genotype"])),
            })
    return results


def load_search_candidates(search_dir: Path) -> list[dict]:
    """Load all feasible candidates from `search_log.csv`.

    This gives the guarded selector a better fallback than the Pareto front
    alone when the front has collapsed to degenerate routing.
    """
    log_path = search_dir / "search_log.csv"
    if not log_path.exists():
        return []

    results = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("time_s") == "INFEASIBLE" or not row.get("cascade_flops"):
                continue
            try:
                results.append({
                    "cascade_acc": float(row["cascade_acc"]),
                    "total_params": int(row["total_params"]),
                    "cascade_flops": float(row["cascade_flops"]),
                    "little_flops": float(row["little_flops"]),
                    "big_flops": float(row["big_flops"]),
                    "exit_ratio": float(row["exit_ratio"]),
                    "total_bytes": int(row["total_bytes"]),
                    "genotype": PairGenotype.from_dict(ast.literal_eval(row["genotype"])),
                })
            except (KeyError, TypeError, ValueError, SyntaxError):
                continue
    return _dedupe_candidates(results)


def write_selected_architectures(
    path: Path,
    picks: list[dict],
    selection_rule: str,
    candidate_source: str,
    reason: str = "",
) -> None:
    """Persist final retrain picks so plots can mark them exactly."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "selection_rule", "candidate_source", "reason",
            "cascade_acc", "exit_ratio", "cascade_flops",
            "total_params", "total_bytes",
            "little_flops", "big_flops", "genotype",
        ])
        for fallback_rank, row in enumerate(picks, start=1):
            rank = row.get("_selection_rank", fallback_rank)
            genotype = row.get("genotype")
            writer.writerow([
                rank,
                selection_rule,
                candidate_source,
                reason,
                f"{row['cascade_acc']:.4f}",
                f"{row['exit_ratio']:.4f}" if row.get("exit_ratio") is not None else "",
                f"{row['cascade_flops']:.0f}" if row.get("cascade_flops") is not None else "",
                row.get("total_params", ""),
                row.get("total_bytes", ""),
                f"{row['little_flops']:.0f}" if row.get("little_flops") is not None else "",
                f"{row['big_flops']:.0f}" if row.get("big_flops") is not None else "",
                str(genotype.to_dict()) if genotype is not None else "",
            ])


def _write_sweep_csv(
    csv_path: Path,
    thresholds: list[float],
    sweep_results: list[dict],
    little_flops: int,
    big_flops: int,
) -> None:
    """Write per-threshold sweep results to CSV.

    Columns: threshold, cascade_acc, exit_ratio, little_acc, big_acc,
             little_flops, big_flops, cascade_flops,
             little_ece, routing_error_rate.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold", "cascade_acc", "exit_ratio",
            "little_acc", "big_acc",
            "little_flops", "big_flops", "cascade_flops",
            "little_ece", "routing_error_rate",
        ])
        for t, r in zip(thresholds, sweep_results):
            cflops = cascade_flops(little_flops, big_flops, r["exit_ratio"])
            writer.writerow([
                f"{t:.2f}",
                f"{r['cascade_acc']:.4f}",
                f"{r['exit_ratio']:.4f}",
                f"{r['little_acc']:.4f}",
                f"{r['big_acc']:.4f}",
                little_flops, big_flops, f"{cflops:.0f}",
                f"{r['little_ece']:.4f}",
                f"{r['routing_error_rate']:.4f}",
            ])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", type=str, required=True, help="Path to NAS search experiment dir")
    parser.add_argument("--top-k", type=int, default=3, help="Number of architectures to retrain")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument(
        "--selection",
        choices=["balanced", "knee", "top_acc"],
        default="balanced",
        help="Selection rule: 'balanced' (default) uses exit-ratio/budget guards; "
             "'knee' picks knee + neighbours; 'top_acc' is a sensitivity check.",
    )
    parser.add_argument(
        "--candidate-source",
        choices=["auto", "pareto", "search_log"],
        default="auto",
        help="Candidate pool for balanced/top_acc selection. auto uses search_log when present.",
    )
    parser.add_argument(
        "--knee-low", type=float, default=0.2,
        help="Lower bound of acceptable exit_ratio for knee-region warning.",
    )
    parser.add_argument(
        "--knee-high", type=float, default=0.8,
        help="Upper bound of acceptable exit_ratio for knee-region warning.",
    )
    parser.add_argument(
        "--min-params", type=int, default=50_000,
        help="Hard minimum parameter count for final selection. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-params", type=int, default=PARAM_BUDGET_REFERENCE,
        help="Hard parameter cap for final selection. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-bytes", type=int, default=MEMORY_BUDGET_BYTES,
        help="Hard deployment-memory cap for final selection. Use 0 to disable.",
    )
    parser.add_argument(
        "--acc-tolerance", type=float, default=0.02,
        help="Absolute proxy-accuracy tolerance for balanced diverse picks.",
    )
    parser.add_argument(
        "--allow-big-flops-le-little",
        action="store_true",
        help="Allow selected cascades where big FLOPs are <= little FLOPs.",
    )
    parser.add_argument(
        "--allow-cascade-flops-ge-big",
        action="store_true",
        help="Allow selected cascades where expected cascade FLOPs are >= big-model FLOPs.",
    )
    parser.add_argument(
        "--only-selection-ranks",
        default="",
        help=(
            "Comma-separated selected architecture ranks to retrain after selection, "
            "for example '2' or '1,3'. Ranks refer to the selected top-k list."
        ),
    )
    args = parser.parse_args()

    default_cfg = Path(__file__).resolve().parent.parent / "configs" / "cifar10_retrain_cascade_only.yaml"
    config = load_config(Path(args.config) if args.config else default_cfg)
    set_seed(config["seed"])
    device = get_device()
    print(f"Device: {device}")

    threshold_sweep = config.get("evaluation", {}).get("threshold_sweep", DEFAULT_THRESHOLDS)
    do_temp_scaling = config.get("evaluation", {}).get("temperature_scaling", False)

    train_loader, val_loader, test_loader = get_data_loaders(
        dataset=config["data"]["dataset"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        seed=config["seed"],
    )

    search_dir = Path(args.search_dir)
    pareto = load_pareto_front(search_dir)
    print(f"Loaded {len(pareto)} Pareto-optimal architectures")

    if args.selection == "knee":
        log_candidates = load_search_candidates(search_dir)
        if args.candidate_source == "pareto":
            pool = pareto
            source = "pareto"
        elif args.candidate_source == "search_log":
            pool = log_candidates
            source = "search_log"
        else:
            pool = log_candidates or pareto
            source = "search_log" if log_candidates else "pareto"

        min_params = None if args.min_params == 0 else args.min_params
        max_params = None if args.max_params == 0 else args.max_params
        max_bytes = None if args.max_bytes == 0 else args.max_bytes
        filtered = filter_budget_feasible(
            pool,
            min_params=min_params,
            max_params=max_params,
            max_bytes=max_bytes,
            require_big_flops_gt_little=not args.allow_big_flops_le_little,
            require_cascade_flops_lt_big=not args.allow_cascade_flops_ge_big,
        )
        frontier = non_dominated_acc_flops(filtered)
        print(
            f"[selection] Candidate source: {source}; "
            f"{len(filtered)}/{len(pool)} pass params>={min_params}, "
            f"params<={max_params}, and bytes<={max_bytes}; "
            f"big_flops>little_flops={not args.allow_big_flops_le_little}; "
            f"cascade_flops<big_flops={not args.allow_cascade_flops_ge_big}; "
            f"{len(frontier)} are non-dominated."
        )
        picks = select_knee_neighbors(frontier, k=args.top_k)
        candidate_source = f"{source}_frontier"
        selection_reason = "knee + nearest FLOPs neighbours"
        # Surface degeneracy of the chosen window so a broken search isn't laundered.
        ers = [p.get("exit_ratio") for p in picks]
        if all(er is None for er in ers):
            print("[selection] WARNING: exit_ratio unavailable on this front - "
                  "cannot verify knee region.")
        elif all((er is not None) and not (args.knee_low <= er <= args.knee_high) for er in ers):
            print(f"[selection] WARNING: every chosen architecture has "
                  f"exit_ratio outside [{args.knee_low}, {args.knee_high}]. "
                  "Search likely converged to degenerate cascades. "
                  "Investigate before retraining.")
    else:
        log_candidates = load_search_candidates(search_dir)
        if args.candidate_source == "pareto":
            pool = pareto
            source = "pareto"
        elif args.candidate_source == "search_log":
            pool = log_candidates
            source = "search_log"
        else:
            pool = log_candidates or pareto
            source = "search_log" if log_candidates else "pareto"
        candidate_source = source

        min_params = None if args.min_params == 0 else args.min_params
        max_params = None if args.max_params == 0 else args.max_params
        max_bytes = None if args.max_bytes == 0 else args.max_bytes
        filtered = filter_budget_feasible(
            pool,
            min_params=min_params,
            max_params=max_params,
            max_bytes=max_bytes,
            require_big_flops_gt_little=not args.allow_big_flops_le_little,
            require_cascade_flops_lt_big=not args.allow_cascade_flops_ge_big,
        )
        print(
            f"[selection] Candidate source: {source}; "
            f"{len(filtered)}/{len(pool)} pass params>={min_params}, "
            f"params<={max_params}, bytes<={max_bytes}, and "
            f"big_flops>little_flops={not args.allow_big_flops_le_little}, "
            f"cascade_flops<big_flops={not args.allow_cascade_flops_ge_big}."
        )
        if args.selection == "balanced":
            picks, reason = select_balanced_candidates(
                filtered,
                k=args.top_k,
                exit_low=args.knee_low,
                exit_high=args.knee_high,
                acc_tolerance=args.acc_tolerance,
            )
            if "fell back" in reason:
                print(
                    f"[selection] WARNING: no candidates have exit_ratio in "
                    f"[{args.knee_low}, {args.knee_high}]. {reason}."
                )
            else:
                print(f"[selection] {reason}.")
            selection_reason = reason
        else:
            picks = sorted(filtered, key=lambda r: r["cascade_acc"], reverse=True)[:args.top_k]
            selection_reason = "top proxy accuracy under hard budgets"

    for idx, pick in enumerate(picks, start=1):
        pick["_selection_rank"] = idx

    selected_ranks = None
    if args.only_selection_ranks.strip():
        selected_ranks = {
            int(part.strip())
            for part in args.only_selection_ranks.split(",")
            if part.strip()
        }
        picks = [pick for pick in picks if pick["_selection_rank"] in selected_ranks]
        if not picks:
            raise ValueError(
                f"--only-selection-ranks={args.only_selection_ranks!r} matched no "
                "selected architectures."
            )

    top_k = len(picks)
    rank_msg = (
        f" ranks {sorted(selected_ranks)}"
        if selected_ranks is not None
        else ""
    )
    print(
        f"Selection rule: {args.selection!r} -> {top_k} architectures{rank_msg} "
        "will be retrained."
    )
    for pick in picks:
        idx = pick["_selection_rank"]
        print(
            f"  Pick {idx}: proxy={pick['cascade_acc']:.4f}, "
            f"exit={pick.get('exit_ratio') if pick.get('exit_ratio') is not None else 'NA'}, "
            f"flops={pick.get('cascade_flops'):.0f}, "
            f"params={pick.get('total_params')}, bytes={pick.get('total_bytes')}"
        )

    exp_dir = get_experiment_dir(config["experiment_name"])
    search_selection_path = search_dir / "selected_architectures.csv"
    exp_selection_path = exp_dir / "selected_architectures.csv"
    write_selected_architectures(
        search_selection_path, picks, args.selection, candidate_source, selection_reason
    )
    write_selected_architectures(
        exp_selection_path, picks, args.selection, candidate_source, selection_reason
    )
    print(f"[selection] Wrote selected architectures -> {search_selection_path}")
    print(f"[selection] Wrote selected architectures -> {exp_selection_path}")

    for local_i, entry in enumerate(picks, start=1):
        selection_rank = entry.get("_selection_rank", local_i)
        genotype = entry["genotype"]
        memory = estimate_pair_memory(genotype)
        flops = estimate_pair_flops(genotype)
        print(f"\n{'='*60}")
        print(f"Retraining selected architecture rank {selection_rank} ({local_i}/{top_k})")
        print(f"Proxy cascade acc: {entry['cascade_acc']:.4f}, Memory: {memory['total_bytes']:,}B")
        print(f"Genotype: {genotype.to_dict()}")
        print(f"{'='*60}\n")

        model = CascadePair(genotype)
        n_params = count_parameters(model)
        print(f"Parameters: {n_params:,}")

        model_name = f"nas_arch_{selection_rank}"

        result = train_joint_model(
            model, train_loader, val_loader, config, device, exp_dir,
            model_name=model_name,
            threshold=genotype.threshold,
        )

        # Reload best checkpoint
        model.load_state_dict(
            torch.load(exp_dir / f"{model_name}_best.pt", map_location=device, weights_only=True)
        )
        model = model.to(device)

        # Sweep thresholds in a single forward pass on the test set.
        sweep = evaluate_at_thresholds(
            model, test_loader, device, thresholds=threshold_sweep, temperature=1.0,
        )
        sweep_csv = exp_dir / f"{model_name}_threshold_sweep.csv"
        _write_sweep_csv(sweep_csv, threshold_sweep, sweep,
                         flops["little_flops"], flops["big_flops"])
        print(f"Wrote threshold sweep -> {sweep_csv}")

        learned_T = None
        if do_temp_scaling:
            learned_T = learn_temperature(model, val_loader, device)
            sweep_T = evaluate_at_thresholds(
                model, test_loader, device,
                thresholds=threshold_sweep, temperature=learned_T,
            )
            sweep_T_csv = exp_dir / f"{model_name}_threshold_sweep_tempscaled.csv"
            _write_sweep_csv(sweep_T_csv, threshold_sweep, sweep_T,
                             flops["little_flops"], flops["big_flops"])
            print(f"Wrote temp-scaled sweep -> {sweep_T_csv} (T={learned_T:.4f})")

        # Human-readable summary.
        with open(exp_dir / f"{model_name}_summary.txt", "w") as f:
            f.write(f"Model: NAS Architecture {selection_rank}\n")
            f.write(f"Genotype: {genotype.to_dict()}\n")
            f.write(f"Parameters: {n_params:,}\n")
            f.write(f"Memory: {memory['total_bytes']:,} bytes\n")
            f.write(f"Little FLOPs: {flops['little_flops']:,}\n")
            f.write(f"Big FLOPs: {flops['big_flops']:,}\n")
            f.write(f"Proxy cascade acc: {entry['cascade_acc']:.4f}\n")
            f.write(f"Best val cascade acc: {result['best_cascade_acc']:.4f}\n")
            if learned_T is not None:
                f.write(f"Learned temperature: {learned_T:.4f}\n")
            f.write("\n")
            f.write("Threshold sweep (test set, T=1.0):\n")
            for t, r in zip(threshold_sweep, sweep):
                cflops = cascade_flops(flops["little_flops"], flops["big_flops"], r["exit_ratio"])
                f.write(
                    f"  t={t:.2f}  cascade_acc={r['cascade_acc']:.4f}  "
                    f"exit={r['exit_ratio']:.2f}  cascade_flops={cflops:,.0f}  "
                    f"ECE={r['little_ece']:.4f}  routing_err={r['routing_error_rate']:.4f}\n"
                )

        # Console preview
        print("\nTest results (T=1.0):")
        for t, r in zip(threshold_sweep, sweep):
            cflops = cascade_flops(flops["little_flops"], flops["big_flops"], r["exit_ratio"])
            print(
                f"  t={t:.2f} | Cascade {r['cascade_acc']:.4f} | Exit {r['exit_ratio']:.2f} | "
                f"Flops {cflops:>10,.0f} | ECE {r['little_ece']:.4f} | "
                f"RoutErr {r['routing_error_rate']:.4f}"
            )


if __name__ == "__main__":
    main()
