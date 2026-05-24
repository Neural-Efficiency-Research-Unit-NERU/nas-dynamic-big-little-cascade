#!/usr/bin/env python
"""Run NSGA-II edge-constrained NAS."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_data_loaders
from src.nas.evolutionary import run_search
from src.training.utils import (
    MEMORY_BUDGET_BYTES,
    get_device,
    get_experiment_dir,
    load_config,
    set_seed,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    default_cfg = Path(__file__).resolve().parent.parent / "configs" / "cifar10_cosearch.yaml"
    config = load_config(Path(args.config) if args.config else default_cfg)
    set_seed(config["seed"])
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_data_loaders(
        dataset=config["data"]["dataset"],
        batch_size=config["search"]["proxy_batch_size"],
        num_workers=config["data"]["num_workers"],
        seed=config["seed"],
    )

    exp_dir = get_experiment_dir(config["experiment_name"])
    memory_budget = config["search"].get("memory_budget_bytes", MEMORY_BUDGET_BYTES)
    param_ref = config["search"].get("param_budget_reference",
                                     config["search"].get("param_budget", 100_000))
    max_params = config["search"].get("max_params", param_ref)
    min_params = config["search"].get("min_params")
    little_min = config["search"].get("little_min_params")
    little_max = config["search"].get("little_max_params")
    big_min = config["search"].get("big_min_params")
    big_max = config["search"].get("big_max_params")
    require_big_more_blocks = config["search"].get("require_big_more_blocks", True)
    require_big_at_least_little_params = config["search"].get(
        "require_big_at_least_little_params", False
    )
    exit_low = config["search"].get("exit_ratio_min", 0.2)
    exit_high = config["search"].get("exit_ratio_max", 0.8)
    exit_penalty = config["search"].get("exit_penalty_weight", 0.10)

    print(f"Experiment dir: {exp_dir}")
    print(f"Population: {config['search']['population_size']}, Generations: {config['search']['n_generations']}")
    print(f"Proxy training: {config['search']['proxy_epochs']} epochs @ LR {config['search']['proxy_lr']}")
    print(f"Hard constraint: deployment memory <= {memory_budget:,} bytes "
          f"({memory_budget / 1024:.0f} KiB)")
    if max_params is None or max_params == 0:
        print("Hard constraint: total params disabled")
    else:
        if min_params:
            print(f"Hard constraint: total params >= {min_params:,}")
        print(f"Hard constraint: total params <= {max_params:,}")
    if little_min:
        print(f"Hard constraint: little params >= {little_min:,}")
    if little_max:
        print(f"Hard constraint: little params <= {little_max:,}")
    if big_min:
        print(f"Hard constraint: big params >= {big_min:,}")
    if big_max:
        print(f"Hard constraint: big params <= {big_max:,}")
    print(f"Hard constraint: big blocks > little blocks = {require_big_more_blocks}")
    print(
        "Hard constraint: big params >= little params = "
        f"{require_big_at_least_little_params}"
    )
    print(f"Param reference: {param_ref:,}")
    print(
        f"Exit-ratio guard: target [{exit_low:.2f}, {exit_high:.2f}], "
        f"objective penalty weight {exit_penalty:.3f}"
    )
    print(f"Objectives: maximise cascade accuracy, minimise avg cascade FLOPs")
    print()

    run_search(train_loader, val_loader, config, device, exp_dir)


if __name__ == "__main__":
    main()
