"""NSGA-II evolutionary NAS with deployment-memory and parameter budgets."""
import csv
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from src.nas.evaluator import proxy_train
from src.nas.search_space import N_VAR, XL, XU, vector_to_pair_genotype
from src.training.utils import (
    MEMORY_BUDGET_BYTES,
    PARAM_BUDGET_REFERENCE,
    budget_utilisation,
    estimate_pair_flops,
    estimate_pair_memory,
)

# Re-exported for backwards compatibility with older scripts/tests.
PARAM_BUDGET = PARAM_BUDGET_REFERENCE


def _is_analytically_feasible(
    genotype,
    memory_budget_bytes: int,
    min_params: int | None,
    max_params: int | None,
    little_min_params: int | None = None,
    little_max_params: int | None = None,
    big_min_params: int | None = None,
    big_max_params: int | None = None,
    require_big_more_blocks: bool = False,
    require_big_at_least_little_params: bool = False,
) -> bool:
    memory = estimate_pair_memory(genotype)
    total_bytes = memory["total_bytes"]
    total_params = memory["total_params"]
    if total_bytes > memory_budget_bytes:
        return False
    if min_params is not None and total_params < min_params:
        return False
    if max_params is not None and total_params > max_params:
        return False
    if little_min_params is not None and memory["little_params"] < little_min_params:
        return False
    if little_max_params is not None and memory["little_params"] > little_max_params:
        return False
    if big_min_params is not None and memory["big_params"] < big_min_params:
        return False
    if big_max_params is not None and memory["big_params"] > big_max_params:
        return False
    if require_big_more_blocks and len(genotype.big_blocks) <= len(genotype.little_blocks):
        return False
    if (
        require_big_at_least_little_params
        and memory["big_params"] < memory["little_params"]
    ):
        return False
    return True


class FeasiblePairSampling(FloatRandomSampling):
    """Initial sampler that avoids analytically infeasible pair genotypes."""

    def __init__(
        self,
        memory_budget_bytes: int,
        min_params: int | None,
        max_params: int | None,
        little_min_params: int | None = None,
        little_max_params: int | None = None,
        big_min_params: int | None = None,
        big_max_params: int | None = None,
        require_big_more_blocks: bool = False,
        require_big_at_least_little_params: bool = False,
        fixed_threshold: float | None = None,
        max_attempts_factor: int = 200,
        verbose: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.memory_budget_bytes = memory_budget_bytes
        self.min_params = min_params
        self.max_params = max_params
        self.little_min_params = little_min_params
        self.little_max_params = little_max_params
        self.big_min_params = big_min_params
        self.big_max_params = big_max_params
        self.require_big_more_blocks = require_big_more_blocks
        self.require_big_at_least_little_params = require_big_at_least_little_params
        self.fixed_threshold = fixed_threshold
        self.max_attempts_factor = max_attempts_factor
        self.verbose = verbose
        self.fallback_random_state = np.random.default_rng(seed)

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        if random_state is None:
            random_state = self.fallback_random_state

        accepted = []
        attempts = 0
        max_attempts = max(n_samples * self.max_attempts_factor, n_samples)

        while len(accepted) < n_samples and attempts < max_attempts:
            attempts += 1
            x = super()._do(problem, 1, *args, random_state=random_state, **kwargs)[0]
            genotype = vector_to_pair_genotype(
                x.tolist(), threshold_override=self.fixed_threshold
            )
            if _is_analytically_feasible(
                genotype,
                self.memory_budget_bytes,
                self.min_params,
                self.max_params,
                self.little_min_params,
                self.little_max_params,
                self.big_min_params,
                self.big_max_params,
                self.require_big_more_blocks,
                self.require_big_at_least_little_params,
            ):
                accepted.append(x)

        if len(accepted) < n_samples:
            raise RuntimeError(
                "Could not sample enough feasible NAS candidates "
                f"({len(accepted)}/{n_samples}) after {attempts} attempts. "
                "Relax min_params/max_params/memory_budget or widen the search space."
            )

        if self.verbose:
            print(
                f"Feasible initial sampling: accepted {len(accepted)}/{attempts} "
                "random candidates"
            )
        return np.asarray(accepted, dtype=float)


class FeasiblePairRepair(Repair):
    """Replace infeasible co-search offspring before expensive proxy training."""

    def __init__(
        self,
        memory_budget_bytes: int,
        min_params: int | None,
        max_params: int | None,
        little_min_params: int | None = None,
        little_max_params: int | None = None,
        big_min_params: int | None = None,
        big_max_params: int | None = None,
        require_big_more_blocks: bool = False,
        require_big_at_least_little_params: bool = False,
        fixed_threshold: float | None = None,
        max_attempts_factor: int = 200,
        seed: int | None = None,
    ):
        super().__init__()
        self.memory_budget_bytes = memory_budget_bytes
        self.min_params = min_params
        self.max_params = max_params
        self.little_min_params = little_min_params
        self.little_max_params = little_max_params
        self.big_min_params = big_min_params
        self.big_max_params = big_max_params
        self.require_big_more_blocks = require_big_more_blocks
        self.require_big_at_least_little_params = require_big_at_least_little_params
        self.fixed_threshold = fixed_threshold
        self.sampler = FeasiblePairSampling(
            memory_budget_bytes=memory_budget_bytes,
            min_params=min_params,
            max_params=max_params,
            little_min_params=little_min_params,
            little_max_params=little_max_params,
            big_min_params=big_min_params,
            big_max_params=big_max_params,
            require_big_more_blocks=require_big_more_blocks,
            require_big_at_least_little_params=require_big_at_least_little_params,
            fixed_threshold=fixed_threshold,
            max_attempts_factor=max_attempts_factor,
            verbose=False,
            seed=seed,
        )

    def _do(self, problem, X, **kwargs):
        repaired = np.asarray(X, dtype=float).copy()
        replacements = 0
        random_state = kwargs.get("random_state")

        for i, x in enumerate(repaired):
            genotype = vector_to_pair_genotype(
                x.tolist(), threshold_override=self.fixed_threshold
            )
            if _is_analytically_feasible(
                genotype,
                self.memory_budget_bytes,
                self.min_params,
                self.max_params,
                self.little_min_params,
                self.little_max_params,
                self.big_min_params,
                self.big_max_params,
                self.require_big_more_blocks,
                self.require_big_at_least_little_params,
            ):
                continue

            repaired[i] = self.sampler._do(
                problem, 1, random_state=random_state,
            )[0]
            replacements += 1

        if replacements:
            print(
                f"Repaired {replacements}/{len(repaired)} infeasible "
                "co-search offspring before evaluation"
            )
        return repaired


class NASProblem(Problem):
    """Multi-objective NSGA-II for the (little, big) cascade pair.

    Objectives (both minimised):
      - f1 = -(cascade_accuracy - exit_ratio_penalty)
      - f2 = avg_cascade_flops

    Hard constraints:
      - total_bytes <= MEMORY_BUDGET_BYTES
        Implements deployment memory: weights + max(peak activations) + input.
      - total_params >= min_params                        (May 2026 guardrail)
      - total_params <= max_params                        (May 2026 guardrail)
    """

    def __init__(self, train_loader, val_loader, proxy_epochs: int, proxy_lr: float,
                 device, log_path: Path | None = None,
                 label_smoothing: float = 0.0,
                 little_loss_type: str = "ce", focal_gamma: float = 2.0,
                 cascade_aware: bool = False,
                 routing_sharpness: float = 20.0,
                 little_aux_weight: float = 0.5,
                 big_aux_weight: float = 0.0,
                 defer_cost_weight: float = 0.01,
                 confident_wrong_weight: float = 0.1,
                 detach_routing_weight: bool = True,
                 fixed_threshold: float | None = None,
                 memory_budget_bytes: int = MEMORY_BUDGET_BYTES,
                 min_params: int | None = None,
                 max_params: int | None = PARAM_BUDGET_REFERENCE,
                 little_min_params: int | None = None,
                 little_max_params: int | None = None,
                 big_min_params: int | None = None,
                 big_max_params: int | None = None,
                 require_big_more_blocks: bool = False,
                 require_big_at_least_little_params: bool = False,
                 exit_ratio_min: float = 0.2,
                 exit_ratio_max: float = 0.8,
                 exit_penalty_weight: float = 0.10):
        super().__init__(
            n_var=N_VAR, n_obj=2, n_ieq_constr=9,
            xl=np.array(XL, dtype=float), xu=np.array(XU, dtype=float),
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.proxy_epochs = proxy_epochs
        self.proxy_lr = proxy_lr
        self.device = device
        self.log_path = log_path
        self.label_smoothing = label_smoothing
        self.little_loss_type = little_loss_type
        self.focal_gamma = focal_gamma
        self.cascade_aware = cascade_aware
        self.routing_sharpness = routing_sharpness
        self.little_aux_weight = little_aux_weight
        self.big_aux_weight = big_aux_weight
        self.defer_cost_weight = defer_cost_weight
        self.confident_wrong_weight = confident_wrong_weight
        self.detach_routing_weight = detach_routing_weight
        self.fixed_threshold = fixed_threshold
        self.memory_budget_bytes = memory_budget_bytes
        self.min_params = min_params
        self.max_params = max_params
        self.little_min_params = little_min_params
        self.little_max_params = little_max_params
        self.big_min_params = big_min_params
        self.big_max_params = big_max_params
        self.require_big_more_blocks = require_big_more_blocks
        self.require_big_at_least_little_params = require_big_at_least_little_params
        self.exit_ratio_min = exit_ratio_min
        self.exit_ratio_max = exit_ratio_max
        self.exit_penalty_weight = exit_penalty_weight
        self.eval_count = 0
        self.generation = 0
        self.metrics_by_genotype: dict[str, dict] = {}

    def _evaluate(self, X, out, *args, **kwargs):
        f1_list, f2_list, g_list = [], [], []
        for x in X:
            self.eval_count += 1
            genotype = vector_to_pair_genotype(
                x.tolist(), threshold_override=self.fixed_threshold
            )
            genotype_key = str(genotype.to_dict())

            # Pre-compute analytical memory + FLOPs (cheap, avoids building infeasible)
            memory = estimate_pair_memory(genotype)
            flops_breakdown = estimate_pair_flops(genotype)
            total_bytes = memory["total_bytes"]
            total_params = memory["total_params"]
            little_params = memory["little_params"]
            big_params = memory["big_params"]
            little_flops = flops_breakdown["little_flops"]
            big_flops = flops_breakdown["big_flops"]
            utilisation = budget_utilisation(total_bytes, self.memory_budget_bytes)

            threshold = genotype.threshold
            n_little = len(genotype.little_blocks)
            n_big = len(genotype.big_blocks)
            byte_over = float(total_bytes - self.memory_budget_bytes)
            param_under = 0.0 if self.min_params is None else float(self.min_params - total_params)
            param_over = 0.0 if self.max_params is None else float(total_params - self.max_params)
            little_param_under = (
                0.0 if self.little_min_params is None
                else float(self.little_min_params - little_params)
            )
            little_param_over = (
                0.0 if self.little_max_params is None
                else float(little_params - self.little_max_params)
            )
            big_param_under = (
                0.0 if self.big_min_params is None
                else float(self.big_min_params - big_params)
            )
            big_param_over = (
                0.0 if self.big_max_params is None
                else float(big_params - self.big_max_params)
            )
            block_order_violation = 0.0
            if self.require_big_more_blocks and n_big <= n_little:
                block_order_violation = float(n_little - n_big + 1)
            param_order_violation = 0.0
            if self.require_big_at_least_little_params and big_params < little_params:
                param_order_violation = float(little_params - big_params)

            if (
                byte_over > 0
                or param_under > 0
                or param_over > 0
                or little_param_under > 0
                or little_param_over > 0
                or big_param_under > 0
                or big_param_over > 0
                or block_order_violation > 0
                or param_order_violation > 0
            ):
                # Skip training for infeasible candidates.
                # Use float("inf") on the flops objective so it cannot accidentally
                # appear on the Pareto front.
                f1_list.append(0.0)
                f2_list.append(float("inf"))
                g_list.append([
                    byte_over, param_under, param_over,
                    little_param_under, little_param_over,
                    big_param_under, big_param_over, block_order_violation,
                    param_order_violation,
                ])
                reasons = []
                if byte_over > 0:
                    reasons.append(f"bytes {total_bytes:,} > {self.memory_budget_bytes:,}")
                if param_under > 0:
                    reasons.append(f"params {total_params:,} < {self.min_params:,}")
                if param_over > 0:
                    reasons.append(f"params {total_params:,} > {self.max_params:,}")
                if little_param_under > 0:
                    reasons.append(f"little params {little_params:,} < {self.little_min_params:,}")
                if little_param_over > 0:
                    reasons.append(f"little params {little_params:,} > {self.little_max_params:,}")
                if big_param_under > 0:
                    reasons.append(f"big params {big_params:,} < {self.big_min_params:,}")
                if big_param_over > 0:
                    reasons.append(f"big params {big_params:,} > {self.big_max_params:,}")
                if block_order_violation > 0:
                    reasons.append(f"big blocks {n_big} <= little blocks {n_little}")
                if param_order_violation > 0:
                    reasons.append(
                        f"big params {big_params:,} < little params {little_params:,}"
                    )
                print(
                    f"  Eval {self.eval_count:3d} | Gen {self.generation:2d} | "
                    f"INFEASIBLE ({'; '.join(reasons)} | "
                    f"util {utilisation:.2f}) | Blocks {n_little}+{n_big}"
                )
                if self.log_path:
                    with open(self.log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            self.eval_count, self.generation,
                            "0.0000", "0.0000", "0.0000", "0.0000", "0.00",
                            "0.0000", "0.0000", "0.0000",
                            total_params, total_bytes, f"{utilisation:.4f}",
                            "0",
                            little_flops, big_flops, "",
                            f"{threshold:.3f}",
                            n_little, n_big,
                            "INFEASIBLE",
                            str(genotype.to_dict()),
                        ])
                continue

            t0 = time.time()
            result = proxy_train(
                genotype, self.train_loader, self.val_loader,
                epochs=self.proxy_epochs, lr=self.proxy_lr, device=self.device,
                label_smoothing=self.label_smoothing,
                threshold=threshold,
                little_loss_type=self.little_loss_type,
                focal_gamma=self.focal_gamma,
                cascade_aware=self.cascade_aware,
                routing_sharpness=self.routing_sharpness,
                little_aux_weight=self.little_aux_weight,
                big_aux_weight=self.big_aux_weight,
                defer_cost_weight=self.defer_cost_weight,
                confident_wrong_weight=self.confident_wrong_weight,
                detach_routing_weight=self.detach_routing_weight,
            )
            elapsed = time.time() - t0

            cascade_acc = result["cascade_acc"]
            exit_ratio = result["exit_ratio"]
            cascade_flops_val = float(result["cascade_flops"])
            exit_distance = max(
                self.exit_ratio_min - exit_ratio,
                exit_ratio - self.exit_ratio_max,
                0.0,
            )
            exit_penalty = self.exit_penalty_weight * exit_distance
            objective_acc = cascade_acc - exit_penalty

            f1_list.append(-objective_acc)
            f2_list.append(cascade_flops_val)
            g_list.append([
                byte_over, param_under, param_over,
                little_param_under, little_param_over,
                big_param_under, big_param_over, block_order_violation,
                param_order_violation,
            ])

            genotype_dict = genotype.to_dict()
            self.metrics_by_genotype[genotype_key] = {
                "cascade_acc": cascade_acc,
                "objective_acc": objective_acc,
                "exit_penalty": exit_penalty,
                "exit_ratio": exit_ratio,
                "cascade_flops": cascade_flops_val,
            }
            print(
                f"  Eval {self.eval_count:3d} | Gen {self.generation:2d} | "
                f"Cascade {cascade_acc:.4f} | Little {result['little_acc']:.4f} | "
                f"Big {result['big_acc']:.4f} | Exit {exit_ratio:.2f} | "
                f"ObjAcc {objective_acc:.4f} | "
                f"ECE {result.get('little_ece', 0):.4f} | "
                f"RoutErr {result.get('routing_error_rate', 0):.4f} | "
                f"Flops {cascade_flops_val:>10,.0f} | "
                f"Bytes {total_bytes:>7,} (util {utilisation:.2f}) | "
                f"Params {total_params:>7,} | Blocks {n_little}+{n_big} | "
                f"Thr {threshold:.3f} | {elapsed:.1f}s"
            )

            if self.log_path:
                with open(self.log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.eval_count, self.generation,
                        f"{cascade_acc:.4f}", f"{objective_acc:.4f}",
                        f"{result['little_acc']:.4f}",
                        f"{result['big_acc']:.4f}", f"{exit_ratio:.2f}",
                        f"{exit_penalty:.4f}",
                        f"{result.get('little_ece', 0):.4f}",
                        f"{result.get('routing_error_rate', 0):.4f}",
                        total_params, total_bytes, f"{utilisation:.4f}",
                        result["n_params"],
                        little_flops, big_flops, f"{cascade_flops_val:.0f}",
                        f"{threshold:.3f}",
                        n_little, n_big,
                        f"{elapsed:.1f}",
                        str(genotype_dict),
                    ])

        self.generation += 1
        out["F"] = np.column_stack([f1_list, f2_list])
        out["G"] = np.array(g_list)


def run_search(
    train_loader,
    val_loader,
    config: dict,
    device,
    experiment_dir: Path,
) -> Path:
    """Run NSGA-II co-search and return path to results CSV.

    Objectives: maximise cascade accuracy, minimise avg cascade FLOPs.
    Hard constraint: deployment memory <= memory_budget_bytes (default 460800).
    """
    pop_size = config["search"]["population_size"]
    n_gen = config["search"]["n_generations"]
    proxy_epochs = config["search"]["proxy_epochs"]
    proxy_lr = config["search"]["proxy_lr"]
    memory_budget_bytes = config["search"].get(
        "memory_budget_bytes", MEMORY_BUDGET_BYTES
    )
    max_params = config["search"].get(
        "max_params",
        config["search"].get(
            "param_budget",
            config["search"].get("param_budget_reference", PARAM_BUDGET_REFERENCE),
        ),
    )
    if max_params is not None and max_params <= 0:
        max_params = None
    min_params = config["search"].get("min_params")
    if min_params is not None and min_params <= 0:
        min_params = None
    little_min_params = config["search"].get("little_min_params")
    if little_min_params is not None and little_min_params <= 0:
        little_min_params = None
    little_max_params = config["search"].get("little_max_params")
    if little_max_params is not None and little_max_params <= 0:
        little_max_params = None
    big_min_params = config["search"].get("big_min_params")
    if big_min_params is not None and big_min_params <= 0:
        big_min_params = None
    big_max_params = config["search"].get("big_max_params")
    if big_max_params is not None and big_max_params <= 0:
        big_max_params = None
    require_big_more_blocks = config["search"].get("require_big_more_blocks", True)
    require_big_at_least_little_params = config["search"].get(
        "require_big_at_least_little_params", False
    )
    exit_ratio_min = config["search"].get("exit_ratio_min", 0.2)
    exit_ratio_max = config["search"].get("exit_ratio_max", 0.8)
    exit_penalty_weight = config["search"].get("exit_penalty_weight", 0.10)
    label_smoothing = config.get("co_training", {}).get("label_smoothing", 0.0)
    little_loss_type = config.get("co_training", {}).get("little_loss_type", "ce")
    focal_gamma = config.get("co_training", {}).get("focal_gamma", 2.0)
    cascade_aware = config.get("co_training", {}).get("cascade_aware", False)
    routing_sharpness = config.get("co_training", {}).get("routing_sharpness", 20.0)
    little_aux_weight = config.get("co_training", {}).get("little_aux_weight", 0.5)
    big_aux_weight = config.get("co_training", {}).get("big_aux_weight", 0.0)
    defer_cost_weight = config.get("co_training", {}).get("defer_cost_weight", 0.01)
    confident_wrong_weight = config.get("co_training", {}).get("confident_wrong_weight", 0.1)
    detach_routing_weight = config.get("co_training", {}).get("detach_routing_weight", True)
    fixed_threshold = config["search"].get("fixed_threshold")
    feasible_sampling_attempts_factor = config["search"].get(
        "feasible_sampling_attempts_factor", 500
    )
    seed = config.get("seed")

    log_path = experiment_dir / "search_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "eval_id", "generation",
            "cascade_acc", "objective_acc",
            "little_acc", "big_acc", "exit_ratio", "exit_penalty",
            "little_ece", "routing_error_rate",
            "total_params", "total_bytes", "budget_utilisation",
            "n_params",
            "little_flops", "big_flops", "cascade_flops",
            "threshold", "n_little_blocks", "n_big_blocks",
            "time_s",
            "genotype",
        ])

    problem = NASProblem(
        train_loader, val_loader, proxy_epochs, proxy_lr, device, log_path,
        label_smoothing=label_smoothing,
        little_loss_type=little_loss_type, focal_gamma=focal_gamma,
        cascade_aware=cascade_aware,
        routing_sharpness=routing_sharpness,
        little_aux_weight=little_aux_weight,
        big_aux_weight=big_aux_weight,
        defer_cost_weight=defer_cost_weight,
        confident_wrong_weight=confident_wrong_weight,
        detach_routing_weight=detach_routing_weight,
        fixed_threshold=fixed_threshold,
        memory_budget_bytes=memory_budget_bytes,
        min_params=min_params,
        max_params=max_params,
        little_min_params=little_min_params,
        little_max_params=little_max_params,
        big_min_params=big_min_params,
        big_max_params=big_max_params,
        require_big_more_blocks=require_big_more_blocks,
        require_big_at_least_little_params=require_big_at_least_little_params,
        exit_ratio_min=exit_ratio_min,
        exit_ratio_max=exit_ratio_max,
        exit_penalty_weight=exit_penalty_weight,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FeasiblePairSampling(
            memory_budget_bytes=memory_budget_bytes,
            min_params=min_params,
            max_params=max_params,
            little_min_params=little_min_params,
            little_max_params=little_max_params,
            big_min_params=big_min_params,
            big_max_params=big_max_params,
            require_big_more_blocks=require_big_more_blocks,
            require_big_at_least_little_params=require_big_at_least_little_params,
            fixed_threshold=fixed_threshold,
            max_attempts_factor=feasible_sampling_attempts_factor,
            seed=seed,
        ),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        repair=FeasiblePairRepair(
            memory_budget_bytes=memory_budget_bytes,
            min_params=min_params,
            max_params=max_params,
            little_min_params=little_min_params,
            little_max_params=little_max_params,
            big_min_params=big_min_params,
            big_max_params=big_max_params,
            require_big_more_blocks=require_big_more_blocks,
            require_big_at_least_little_params=require_big_at_least_little_params,
            fixed_threshold=fixed_threshold,
            max_attempts_factor=feasible_sampling_attempts_factor,
            seed=None if seed is None else seed + 1,
        ),
        eliminate_duplicates=True,
    )

    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)

    # Save Pareto front
    pareto_path = experiment_dir / "pareto_front.csv"
    with open(pareto_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cascade_acc", "objective_acc", "exit_ratio", "exit_penalty",
            "cascade_flops",
            "total_params", "total_bytes", "budget_utilisation",
            "little_flops", "big_flops",
            "threshold", "n_little_blocks", "n_big_blocks",
            "genotype",
        ])
        for x, obj in zip(result.X, result.F):
            genotype = vector_to_pair_genotype(
                x.tolist(), threshold_override=fixed_threshold
            )
            metrics = problem.metrics_by_genotype.get(str(genotype.to_dict()), {})
            mem = estimate_pair_memory(genotype)
            flops = estimate_pair_flops(genotype)
            cascade_acc_val = metrics.get("cascade_acc", -obj[0])
            objective_acc_val = metrics.get("objective_acc", -obj[0])
            exit_ratio_val = metrics.get("exit_ratio", "")
            exit_penalty_val = metrics.get("exit_penalty", "")
            cascade_flops_val = metrics.get("cascade_flops", obj[1])
            writer.writerow([
                f"{cascade_acc_val:.4f}", f"{objective_acc_val:.4f}",
                f"{exit_ratio_val:.4f}" if exit_ratio_val != "" else "",
                f"{exit_penalty_val:.4f}" if exit_penalty_val != "" else "",
                f"{cascade_flops_val:.0f}",
                mem["total_params"], mem["total_bytes"],
                f"{budget_utilisation(mem['total_bytes'], memory_budget_bytes):.4f}",
                flops["little_flops"], flops["big_flops"],
                f"{genotype.threshold:.3f}",
                len(genotype.little_blocks), len(genotype.big_blocks),
                str(genotype.to_dict()),
            ])

    print(f"\nSearch complete. {problem.eval_count} evaluations.")
    print(f"Pareto front: {len(result.F)} solutions")
    print(f"Results saved to: {experiment_dir}")

    return log_path
