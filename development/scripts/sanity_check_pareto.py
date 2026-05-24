#!/usr/bin/env python
"""Sanity-check a saved Pareto front from the previous NAS run.

Derives `exit_ratio` from `(cascade_flops, little_flops, big_flops)` because
the joint-NAS CSV does not store it directly:

    cascade_flops = little_flops + (1 - exit_ratio) * big_flops
    => exit_ratio = 1 - (cascade_flops - little_flops) / big_flops

Reports:
- exit_ratio for every Pareto point
- Whether top-3 by cascade_acc are degenerate (exit_ratio outside [0.2, 0.8])
- Whether at least one Pareto point sits in the knee region

Use this to inspect whether selected Pareto points have reasonable exit ratios
before launching a rerun.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def derive_exit_ratio(cascade_flops: float, little_flops: float, big_flops: float) -> float:
    if big_flops <= 0:
        return float("nan")
    return 1.0 - (cascade_flops - little_flops) / big_flops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto-csv", type=Path, required=True)
    parser.add_argument("--knee-low", type=float, default=0.2)
    parser.add_argument("--knee-high", type=float, default=0.8)
    args = parser.parse_args()

    rows: list[dict] = []
    with open(args.pareto_csv) as f:
        for r in csv.DictReader(f):
            cflops = float(r["cascade_flops"])
            lflops = float(r["little_flops"])
            bflops = float(r["big_flops"])
            rows.append({
                "cascade_acc": float(r["cascade_acc"]),
                "cascade_flops": cflops,
                "little_flops": lflops,
                "big_flops": bflops,
                "threshold": float(r["threshold"]),
                "exit_ratio": derive_exit_ratio(cflops, lflops, bflops),
            })

    rows_by_acc = sorted(rows, key=lambda r: r["cascade_acc"], reverse=True)
    top3 = rows_by_acc[:3]
    in_knee = [r for r in rows if args.knee_low <= r["exit_ratio"] <= args.knee_high]

    print(f"Pareto file: {args.pareto_csv}")
    print(f"Total points: {len(rows)}")
    print(f"Points in knee region [{args.knee_low}, {args.knee_high}]: {len(in_knee)}")
    print()
    print("Top-3 by cascade_acc (legacy selection):")
    for i, r in enumerate(top3, start=1):
        flag = "DEGENERATE" if not (args.knee_low <= r["exit_ratio"] <= args.knee_high) else "OK"
        print(
            f"  {i}. acc={r['cascade_acc']:.4f}  "
            f"cflops={r['cascade_flops']:>10,.0f}  "
            f"exit={r['exit_ratio']:+.3f}  "
            f"thr={r['threshold']:.3f}  [{flag}]"
        )

    print()
    n_degen = sum(1 for r in top3 if not (args.knee_low <= r["exit_ratio"] <= args.knee_high))
    if n_degen >= 2:
        print(f"FINDING: {n_degen}/3 top-by-acc picks are degenerate. "
              "Selection-bias hypothesis confirmed.")
    else:
        print(f"FINDING: Only {n_degen}/3 top-by-acc picks are degenerate. "
              "Investigate further before relying on knee selection alone.")


if __name__ == "__main__":
    main()
