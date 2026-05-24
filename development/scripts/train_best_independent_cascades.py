#!/usr/bin/env python
"""Train independent cascades from best-accuracy and largest searched models.

This is a post-search utility for the independent NAS baseline. It reads
little_search_log.csv and big_search_log.csv, selects two models per role:

  * highest proxy validation accuracy
  * highest parameter count

It then trains every little/big combination from scratch and writes the
threshold-sweep results to best_combined_cascade_results.csv.
"""
import argparse
import ast
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_data_loaders
from src.nas.independent_search import evaluate_combined_cascades
from src.nas.search_space import BlockGene, PairGenotype
from src.training.utils import get_device, get_experiment_dir, load_config, set_seed


def _parse_blocks(raw: str) -> list[BlockGene]:
    blocks = []
    for block in ast.literal_eval(raw):
        blocks.append(BlockGene(
            channels=int(block["ch"]),
            layers=int(block["ly"]),
            kernel_size=int(block["ks"]),
            conv_type=block["ct"],
            use_residual=bool(block["res"]),
        ))
    return blocks


def _load_search_log(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing search log: {path}")

    rows: list[dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "eval_id": int(row["eval_id"]),
                "generation": int(row["generation"]),
                "accuracy": float(row["accuracy"]),
                "params": int(row["params"]),
                "blocks": _parse_blocks(row["genotype_str"]),
            })

    if not rows:
        raise ValueError(f"No candidate rows found in {path}")
    return rows


def _select_accuracy_and_param_extremes(rows: list[dict], role: str) -> list[tuple[str, dict]]:
    highest_acc = max(rows, key=lambda r: (r["accuracy"], r["params"]))
    highest_params = max(rows, key=lambda r: (r["params"], r["accuracy"]))

    selected = [
        (f"{role}_highest_acc", highest_acc),
        (f"{role}_highest_params", highest_params),
    ]

    unique: list[tuple[str, dict]] = []
    seen_eval_ids: set[int] = set()
    for label, row in selected:
        if row["eval_id"] in seen_eval_ids:
            print(
                f"WARNING: {role} {label} is the same candidate as another "
                f"selection (eval_id={row['eval_id']}); skipping duplicate."
            )
            continue
        seen_eval_ids.add(row["eval_id"])
        unique.append((label, row))

    return unique


def _format_blocks(blocks: list[BlockGene]) -> str:
    parts = []
    for i, block in enumerate(blocks, start=1):
        conv = "dw" if block.conv_type == "depthwise_separable" else "std"
        res = "res" if block.use_residual else "nores"
        parts.append(
            f"B{i}(ch={block.channels}, layers={block.layers}, "
            f"k={block.kernel_size}, {conv}, {res})"
        )
    return "; ".join(parts)


def _print_candidate_summary(
    little_selected: list[tuple[str, dict]],
    big_selected: list[tuple[str, dict]],
) -> None:
    print("=" * 60)
    print("Selected independent candidates")
    print("=" * 60)

    for role, selected in (("little", little_selected), ("big", big_selected)):
        print(f"\n{role.upper()} candidates")
        for label, row in selected:
            print(
                f"  {label}: eval={row['eval_id']} gen={row['generation']} "
                f"proxy_acc={row['accuracy']:.4f} params={row['params']:,} "
                f"blocks={len(row['blocks'])}"
            )
            print(f"    {_format_blocks(row['blocks'])}")

    print("\nCascade combinations to train")
    combo_id = 0
    for little_label, little in little_selected:
        for big_label, big in big_selected:
            combo_id += 1
            total_params = little["params"] + big["params"]
            print(
                f"  {combo_id}. {little_label} + {big_label}: "
                f"little={little['params']:,}, big={big['params']:,}, "
                f"total={total_params:,}, blocks={len(little['blocks'])}+{len(big['blocks'])}, "
                f"proxy_accs={little['accuracy']:.4f}/{big['accuracy']:.4f}"
            )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 2x2 best independent cascades from search logs.",
    )
    default_cfg = Path(__file__).resolve().parent.parent / "configs" / "cifar10_independent_search.yaml"
    parser.add_argument("--config", type=str, default=str(default_cfg), help="Path to independent NAS config YAML")
    parser.add_argument(
        "--output",
        type=str,
        default="best_combined_cascade_results.csv",
        help="Output CSV filename inside the independent experiment directory.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    set_seed(config["seed"])
    device = get_device()
    print(f"Device: {device}")

    train_loader, _, test_loader = get_data_loaders(
        dataset=config["data"]["dataset"],
        batch_size=config["search"]["proxy_batch_size"],
        num_workers=config["data"]["num_workers"],
        seed=config["seed"],
    )

    exp_dir = get_experiment_dir(config["experiment_name"])
    little_rows = _load_search_log(exp_dir / "little_search_log.csv")
    big_rows = _load_search_log(exp_dir / "big_search_log.csv")

    little_selected = _select_accuracy_and_param_extremes(little_rows, "little")
    big_selected = _select_accuracy_and_param_extremes(big_rows, "big")
    _print_candidate_summary(little_selected, big_selected)

    genotypes: list[PairGenotype] = []
    metadata: list[dict] = []

    for little_label, little in little_selected:
        for big_label, big in big_selected:
            genotypes.append(PairGenotype(
                little_blocks=little["blocks"],
                big_blocks=big["blocks"],
                threshold=config["combine"].get("threshold", 0.70),
            ))
            metadata.append({
                "selection": f"{little_label}+{big_label}",
                "little_source": little_label,
                "big_source": big_label,
                "little_eval_id": little["eval_id"],
                "big_eval_id": big["eval_id"],
                "little_proxy_acc": f"{little['accuracy']:.4f}",
                "big_proxy_acc": f"{big['accuracy']:.4f}",
                "little_search_params": little["params"],
                "big_search_params": big["params"],
                "little_search_blocks": len(little["blocks"]),
                "big_search_blocks": len(big["blocks"]),
            })

    print(f"Training {len(genotypes)} cascade combinations")
    print("=" * 60)

    co = config.get("co_training", {})
    results_path = evaluate_combined_cascades(
        genotypes,
        train_loader,
        test_loader,
        config["combine"]["thresholds"],
        config["combine"].get("retrain_epochs", config["search"]["proxy_epochs"]),
        config["search"]["proxy_lr"],
        device,
        exp_dir,
        little_loss_type=co.get("little_loss_type", "ce"),
        label_smoothing=co.get("label_smoothing", 0.0),
        focal_gamma=co.get("focal_gamma", 2.0),
        results_filename=args.output,
        genotype_metadata=metadata,
    )
    print(f"\nBest independent cascade results saved to: {results_path}")


if __name__ == "__main__":
    main()
