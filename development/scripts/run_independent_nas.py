#!/usr/bin/env python
"""Run Independent NAS baseline: search little and big separately, then combine."""
import ast
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.data import get_data_loaders
from src.nas.independent_search import (
    run_single_search, combine_pareto_fronts, evaluate_combined_cascades,
)
from src.nas.search_space import BlockGene
from src.training.utils import (
    MEMORY_BUDGET_BYTES,
    get_device,
    get_experiment_dir,
    load_config,
    set_seed,
)


def _load_pareto_front_csv(path: Path) -> list[dict]:
    """Load an independent-search Pareto front written by run_single_search."""
    rows: list[dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            blocks = [
                BlockGene(
                    channels=int(block["ch"]),
                    layers=int(block["ly"]),
                    kernel_size=int(block["ks"]),
                    conv_type=block["ct"],
                    use_residual=bool(block["res"]),
                )
                for block in ast.literal_eval(row["genotype"])
            ]
            rows.append({
                "accuracy": float(row["accuracy"]),
                "params": int(row["params"]),
                "blocks": blocks,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Independent NAS baseline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Reuse existing little/big Pareto CSVs and only run cascade combination/evaluation.",
    )
    args = parser.parse_args()

    default_cfg = Path(__file__).resolve().parent.parent / "configs" / "cifar10_independent_search.yaml"
    config = load_config(Path(args.config) if args.config else default_cfg)
    set_seed(config["seed"])
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_data_loaders(
        dataset=config["data"]["dataset"],
        batch_size=config["search"]["proxy_batch_size"],
        num_workers=config["data"]["num_workers"],
        seed=config["seed"],
    )

    exp_dir = get_experiment_dir(config["experiment_name"])

    if args.combine_only:
        little_path = exp_dir / "little_pareto_front.csv"
        big_path = exp_dir / "big_pareto_front.csv"
        if not little_path.exists() or not big_path.exists():
            raise FileNotFoundError(
                "combine-only requires existing little_pareto_front.csv and "
                f"big_pareto_front.csv in {exp_dir}"
            )
        print("=" * 60)
        print("Reusing existing independent Pareto fronts")
        print(f"  Little: {little_path}")
        print(f"  Big   : {big_path}")
        print("=" * 60)
        little_pareto = _load_pareto_front_csv(little_path)
        big_pareto = _load_pareto_front_csv(big_path)
    else:
        # Phase 1: Search little model
        print("=" * 60)
        print("Phase 1: Searching LITTLE model architecture")
        little_min = config["budget_split"].get("little_min", 0)
        print(
            f"  Budget: {little_min:,}-{config['budget_split']['little']:,} params"
        )
        print("=" * 60)
        little_log, little_pareto = run_single_search(
            "little", train_loader, val_loader, config, device, exp_dir,
        )
        print(f"Little search complete. {len(little_pareto)} Pareto-optimal architectures found.")

        # Phase 2: Search big model
        print("=" * 60)
        print("Phase 2: Searching BIG model architecture")
        big_min = config["budget_split"].get("big_min", 0)
        print(
            f"  Budget: {big_min:,}-{config['budget_split']['big']:,} params"
        )
        print("=" * 60)
        big_log, big_pareto = run_single_search(
            "big", train_loader, val_loader, config, device, exp_dir,
        )
        print(f"Big search complete. {len(big_pareto)} Pareto-optimal architectures found.")

    # Phase 3: Combine and evaluate
    total_budget = config["search"].get("param_budget", 100_000)
    min_total_budget = config["search"].get("min_params", 50_000)
    max_memory_bytes = config["search"].get("memory_budget_bytes", MEMORY_BUDGET_BYTES)
    top_k = config["combine"]["top_k"]
    pair_threshold = config["combine"].get("threshold", 0.70)
    budget_tolerance = config["combine"].get("budget_tolerance", 0.35)
    require_big_more_blocks = config["combine"].get("require_big_more_blocks", True)
    require_big_at_least_little_params = config["combine"].get(
        "require_big_at_least_little_params", True
    )
    save_checkpoints = config["combine"].get("save_checkpoints", True)
    budget_split = config.get("budget_split", {})
    retrain_epochs = config["combine"].get(
        "retrain_epochs", config["search"]["proxy_epochs"]
    )
    thresholds = config["combine"]["thresholds"]
    print("=" * 60)
    print("Phase 3: Combining Pareto fronts and evaluating cascades")
    print(f"  Final combined-pair training: {retrain_epochs} epochs")
    print(f"  Require total params: {min_total_budget:,}-{total_budget:,}")
    print(f"  Require deployment memory <= {max_memory_bytes:,} bytes")
    print(f"  Require big blocks > little blocks: {require_big_more_blocks}")
    print(f"  Require big params >= little params: {require_big_at_least_little_params}")
    print(f"  Save independent final checkpoints: {save_checkpoints}")
    print("  Final cascade evaluation: CIFAR-10 test split")
    print("=" * 60)

    genotypes, genotype_metadata = combine_pareto_fronts(
        little_pareto,
        big_pareto,
        total_budget,
        top_k,
        threshold=pair_threshold,
        little_min_budget=budget_split.get("little_min"),
        little_budget=budget_split.get("little"),
        big_min_budget=budget_split.get("big_min"),
        big_budget=budget_split.get("big"),
        budget_tolerance=budget_tolerance,
        require_big_more_blocks=require_big_more_blocks,
        require_big_at_least_little_params=require_big_at_least_little_params,
        min_total_budget=min_total_budget,
        max_memory_bytes=max_memory_bytes,
        return_metadata=True,
    )
    print(f"Generated {len(genotypes)} feasible cascade combinations")

    if len(genotypes) == 0:
        print("WARNING: No feasible combinations found within budget. Try adjusting budget_split.")
        return

    co = config.get("co_training", {})
    results_path = evaluate_combined_cascades(
        genotypes, train_loader, test_loader, thresholds,
        retrain_epochs, config["search"]["proxy_lr"],
        device, exp_dir,
        little_loss_type=co.get("little_loss_type", "ce"),
        label_smoothing=co.get("label_smoothing", 0.0),
        focal_gamma=co.get("focal_gamma", 2.0),
        genotype_metadata=genotype_metadata,
        save_checkpoints=save_checkpoints,
    )
    print(f"\nResults saved to: {results_path}")
    print("Independent NAS baseline complete.")


if __name__ == "__main__":
    main()
