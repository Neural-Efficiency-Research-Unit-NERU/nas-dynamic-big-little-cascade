"""Independent NAS baseline: search little and big models separately, then combine."""
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from src.models.paired_models import ConvBlock
from src.nas.search_space import (
    INPUT_CHANNELS,
    KERNEL_OPTIONS,
    LAYERS_OPTIONS,
    CONV_TYPE_OPTIONS,
    LITTLE_BLOCKS_MIN,
    LITTLE_BLOCKS_MAX,
    LITTLE_CHANNELS,
    BIG_BLOCKS_MIN,
    BIG_BLOCKS_MAX,
    BIG_CHANNELS,
    BlockGene,
    PairGenotype,
)
from src.training.trainer import train_one_epoch, evaluate
from src.training.utils import (
    BYTES_PER_PARAM,
    MEMORY_BUDGET_BYTES,
    _estimate_path_flops,
    _estimate_path_memory,
    cascade_flops,
    compute_ece,
    count_parameters,
    estimate_pair_memory,
)

_GENES_PER_BLOCK = 5


# ---------------------------------------------------------------------------
# 1. Build a standalone model from BlockGene list
# ---------------------------------------------------------------------------

def build_single_model(blocks: list[BlockGene], num_classes: int = 10) -> nn.Sequential:
    """Build a standalone model from a list of BlockGene.

    Same construction as CascadePair's self.little / self.big internally:
    ConvBlock(stride=2) per block, then AdaptiveAvgPool2d(1) + Flatten + Linear.
    """
    layers: list[nn.Module] = []
    in_ch = INPUT_CHANNELS
    for gene in blocks:
        layers.append(ConvBlock(
            in_ch, gene.channels, gene.layers, gene.kernel_size,
            gene.conv_type, gene.use_residual, stride=2,
        ))
        in_ch = gene.channels
    layers.extend([
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(in_ch, num_classes),
    ])
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# 2. Vector encoding / decoding for single-model search
# ---------------------------------------------------------------------------

def decode_single_vector(
    vec: list[float],
    role: str,
    channel_options: list[int],
    blocks_min: int,
    blocks_max: int,
) -> list[BlockGene]:
    """Decode a single-model vector into a list of BlockGene.

    Vector layout:
      [0 .. max_blocks*5-1]  block gene slots (5 per block)
      [max_blocks*5]         n_active (active block count)

    Same decoding logic as _decode_blocks in search_space.py, starting at index 0.
    """
    n_active_raw = vec[blocks_max * _GENES_PER_BLOCK]
    n_active = int(round(max(blocks_min, min(blocks_max, n_active_raw))))

    blocks: list[BlockGene] = []
    for i in range(n_active):
        offset = i * _GENES_PER_BLOCK
        ch_idx = int(round(vec[offset])) % len(channel_options)
        ly_idx = int(round(vec[offset + 1])) % len(LAYERS_OPTIONS)
        ks_idx = int(round(vec[offset + 2])) % len(KERNEL_OPTIONS)
        ct_idx = int(round(vec[offset + 3])) % len(CONV_TYPE_OPTIONS)
        res = vec[offset + 4] >= 0.5
        blocks.append(BlockGene(
            channels=channel_options[ch_idx],
            layers=LAYERS_OPTIONS[ly_idx],
            kernel_size=KERNEL_OPTIONS[ks_idx],
            conv_type=CONV_TYPE_OPTIONS[ct_idx],
            use_residual=res,
        ))
    return blocks


def encode_single_blocks(
    blocks: list[BlockGene],
    channel_options: list[int],
    max_blocks: int,
) -> list[float]:
    """Encode a list of BlockGene into a single-model vector.

    Pads inactive slots by repeating the last active block.
    Appends n_active as the final element.
    """
    vec: list[float] = []
    for i in range(max_blocks):
        b = blocks[i] if i < len(blocks) else blocks[-1]
        vec.extend([
            float(channel_options.index(b.channels)),
            float(LAYERS_OPTIONS.index(b.layers)),
            float(KERNEL_OPTIONS.index(b.kernel_size)),
            float(CONV_TYPE_OPTIONS.index(b.conv_type)),
            1.0 if b.use_residual else 0.0,
        ])
    vec.append(float(len(blocks)))
    return vec


# ---------------------------------------------------------------------------
# 3. SingleModelNASProblem
# ---------------------------------------------------------------------------

def _single_model_bounds(
    channel_options: list[int],
    blocks_min: int,
    blocks_max: int,
) -> tuple[list[float], list[float]]:
    """Compute lower/upper bounds for a single-model search vector."""
    block_xl = [0, 0, 0, 0, 0]
    block_xu = [
        len(channel_options) - 1,
        len(LAYERS_OPTIONS) - 1,
        len(KERNEL_OPTIONS) - 1,
        len(CONV_TYPE_OPTIONS) - 1,
        1,
    ]
    xl = block_xl * blocks_max + [float(blocks_min)]
    xu = block_xu * blocks_max + [float(blocks_max)]
    return xl, xu


def _single_model_param_count(
    vec: list[float],
    role: str,
    channel_options: list[int],
    blocks_min: int,
    blocks_max: int,
) -> int:
    blocks = decode_single_vector(
        vec, role, channel_options, blocks_min, blocks_max,
    )
    weight_bytes, _ = _estimate_path_memory(blocks, INPUT_CHANNELS)
    return weight_bytes // BYTES_PER_PARAM


class FeasibleSingleModelSampling(FloatRandomSampling):
    """Initial sampler that avoids analytically infeasible role candidates."""

    def __init__(
        self,
        role: str,
        channel_options: list[int],
        blocks_min: int,
        blocks_max: int,
        min_param_budget: int,
        param_budget: int,
        max_attempts_factor: int = 500,
        verbose: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.role = role
        self.channel_options = channel_options
        self.blocks_min = blocks_min
        self.blocks_max = blocks_max
        self.min_param_budget = min_param_budget
        self.param_budget = param_budget
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
            n_params = _single_model_param_count(
                x.tolist(), self.role, self.channel_options,
                self.blocks_min, self.blocks_max,
            )

            if self.min_param_budget <= n_params <= self.param_budget:
                accepted.append(x)

        if len(accepted) < n_samples:
            raise RuntimeError(
                f"Could not sample enough feasible independent {self.role} candidates "
                f"({len(accepted)}/{n_samples}) after {attempts} attempts. "
                "Relax budget_split, widen the search space, or increase "
                "search.feasible_sampling_attempts_factor."
            )

        if self.verbose:
            print(
                f"[{self.role}] Feasible initial sampling: accepted "
                f"{len(accepted)}/{attempts} random candidates"
            )
        return np.asarray(accepted, dtype=float)


class FeasibleSingleModelRepair(Repair):
    """Replace infeasible independent-search offspring before evaluation."""

    def __init__(
        self,
        role: str,
        channel_options: list[int],
        blocks_min: int,
        blocks_max: int,
        min_param_budget: int,
        param_budget: int,
        max_attempts_factor: int = 500,
        seed: int | None = None,
    ):
        super().__init__()
        self.role = role
        self.channel_options = channel_options
        self.blocks_min = blocks_min
        self.blocks_max = blocks_max
        self.min_param_budget = min_param_budget
        self.param_budget = param_budget
        self.sampler = FeasibleSingleModelSampling(
            role=role,
            channel_options=channel_options,
            blocks_min=blocks_min,
            blocks_max=blocks_max,
            min_param_budget=min_param_budget,
            param_budget=param_budget,
            max_attempts_factor=max_attempts_factor,
            verbose=False,
            seed=seed,
        )

    def _do(self, problem, X, **kwargs):
        repaired = np.asarray(X, dtype=float).copy()
        replacements = 0
        random_state = kwargs.get("random_state")

        for i, x in enumerate(repaired):
            n_params = _single_model_param_count(
                x.tolist(), self.role, self.channel_options,
                self.blocks_min, self.blocks_max,
            )
            if self.min_param_budget <= n_params <= self.param_budget:
                continue

            repaired[i] = self.sampler._do(
                problem, 1, random_state=random_state,
            )[0]
            replacements += 1

        if replacements:
            print(
                f"[{self.role}] Repaired {replacements}/{len(repaired)} "
                "infeasible offspring before evaluation"
            )
        return repaired


class SingleModelNASProblem(Problem):
    """Multi-objective single-model search: minimize -accuracy, minimize params.

    Hard constraints: min_param_budget <= params <= param_budget.
    """

    def __init__(
        self,
        role: str,
        channel_options: list[int],
        blocks_min: int,
        blocks_max: int,
        param_budget: int,
        min_param_budget: int,
        train_loader,
        val_loader,
        proxy_epochs: int,
        proxy_lr: float,
        device: torch.device,
        log_path: Path | None = None,
        loss_type: str = "ce",
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
    ):
        xl, xu = _single_model_bounds(channel_options, blocks_min, blocks_max)
        n_var = blocks_max * _GENES_PER_BLOCK + 1
        super().__init__(
            n_var=n_var, n_obj=2, n_ieq_constr=2,
            xl=np.array(xl, dtype=float), xu=np.array(xu, dtype=float),
        )
        self.role = role
        self.channel_options = channel_options
        self.blocks_min = blocks_min
        self.blocks_max = blocks_max
        self.param_budget = param_budget
        self.min_param_budget = min_param_budget
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.proxy_epochs = proxy_epochs
        self.proxy_lr = proxy_lr
        self.device = device
        self.log_path = log_path
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.eval_count = 0
        self.generation = 0

    def _evaluate(self, X, out, *args, **kwargs):
        f1_list, f2_list, g_min_list, g_max_list = [], [], [], []
        for x in X:
            self.eval_count += 1
            blocks = decode_single_vector(
                x.tolist(), self.role, self.channel_options,
                self.blocks_min, self.blocks_max,
            )

            # Analytical param estimate (avoids building model for infeasible)
            weight_bytes, _ = _estimate_path_memory(blocks, INPUT_CHANNELS)
            n_params = weight_bytes // 4  # float32

            below_min = n_params < self.min_param_budget
            above_max = n_params > self.param_budget

            if below_min or above_max:
                f1_list.append(0.0)  # worst accuracy (negated: 0)
                f2_list.append(n_params)
                g_min_list.append(self.min_param_budget - n_params)
                g_max_list.append(n_params - self.param_budget)
                reason = (
                    f"params {n_params:,} < {self.min_param_budget:,}"
                    if below_min
                    else f"params {n_params:,} > {self.param_budget:,}"
                )
                print(
                    f"  [{self.role}] Eval {self.eval_count:3d} | Gen {self.generation:2d} | "
                    f"INFEASIBLE ({reason}) | "
                    f"Blocks {len(blocks)}"
                )
                continue

            t0 = time.time()
            result = proxy_train_single(
                blocks, self.train_loader, self.val_loader,
                self.proxy_epochs, self.proxy_lr, self.device,
                loss_type=self.loss_type,
                label_smoothing=self.label_smoothing,
                focal_gamma=self.focal_gamma,
            )
            elapsed = time.time() - t0

            acc = result["accuracy"]
            f1_list.append(-acc)
            f2_list.append(n_params)
            g_min_list.append(self.min_param_budget - n_params)
            g_max_list.append(n_params - self.param_budget)

            print(
                f"  [{self.role}] Eval {self.eval_count:3d} | Gen {self.generation:2d} | "
                f"Acc {acc:.4f} | Params {n_params:>7,} | "
                f"Blocks {len(blocks)} | {elapsed:.1f}s"
            )

            if self.log_path:
                genotype_str = str([
                    {"ch": b.channels, "ly": b.layers, "ks": b.kernel_size,
                     "ct": b.conv_type, "res": b.use_residual}
                    for b in blocks
                ])
                with open(self.log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.eval_count, self.generation,
                        f"{acc:.4f}", n_params,
                        f"{elapsed:.1f}", genotype_str,
                    ])

        self.generation += 1
        out["F"] = np.column_stack([f1_list, f2_list])
        out["G"] = np.column_stack([g_min_list, g_max_list])


# ---------------------------------------------------------------------------
# 4. Proxy training for a single model (CE only)
# ---------------------------------------------------------------------------

def proxy_train_single(
    blocks: list[BlockGene],
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: torch.device,
    loss_type: str = "ce",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> dict:
    """Train a single model with the configured loss. Returns {accuracy, n_params}.

    `loss_type` accepts "ce" or "focal"; `label_smoothing` is applied in either
    branch. This mirrors the joint NAS proxy so RQ1 isolates joint vs.
    independent rather than mixing it with focal vs. CE.
    """
    import torch.nn.functional as F

    from src.training.trainer import focal_loss

    model = build_single_model(blocks).to(device)
    n_params = count_parameters(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            if loss_type == "focal":
                loss = focal_loss(logits, targets, gamma=focal_gamma, label_smoothing=label_smoothing)
            else:
                loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()

    criterion = nn.CrossEntropyLoss()  # eval criterion is reporting-only
    _, accuracy = evaluate(model, val_loader, criterion, device)
    return {"accuracy": accuracy, "n_params": n_params}


# ---------------------------------------------------------------------------
# 5. Run NSGA-II for one role
# ---------------------------------------------------------------------------

def run_single_search(
    role: str,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
    experiment_dir: Path,
) -> tuple[Path, list[dict]]:
    """Run NSGA-II for a single role ('little' or 'big').

    Returns (log_csv_path, pareto_results) where pareto_results is a list of
    dicts with keys: accuracy, params, blocks (list[BlockGene]).
    """
    search_space = config.get("search_space", {})
    if role == "little":
        channel_options = search_space.get("little_channels", LITTLE_CHANNELS)
        blocks_min, blocks_max = search_space.get(
            "little_blocks_range", [LITTLE_BLOCKS_MIN, LITTLE_BLOCKS_MAX]
        )
    elif role == "big":
        channel_options = search_space.get("big_channels", BIG_CHANNELS)
        blocks_min, blocks_max = search_space.get(
            "big_blocks_range", [BIG_BLOCKS_MIN, BIG_BLOCKS_MAX]
        )
    else:
        raise ValueError(f"Unknown role: {role!r}. Expected 'little' or 'big'.")

    pop_size = config["search"]["population_size"]
    n_gen = config["search"]["n_generations"]
    proxy_epochs = config["search"]["proxy_epochs"]
    proxy_lr = config["search"]["proxy_lr"]
    param_budget = config["search"].get("param_budget", 100_000)

    # Per-role sub-budget. Prefer the explicit split used by the experiment
    # configs; keep the legacy search.<role>_param_budget and 50/50 fallbacks.
    budget_split = config.get("budget_split", {})
    sub_budget = budget_split.get(
        role,
        config["search"].get(f"{role}_param_budget", param_budget // 2),
    )
    min_budget_split = config.get("min_budget_split", {})
    sub_min_budget = budget_split.get(
        f"{role}_min",
        min_budget_split.get(role, config["search"].get(f"{role}_min_param_budget", 0)),
    )
    sub_min_budget = 0 if sub_min_budget is None else int(sub_min_budget)
    sub_budget = int(sub_budget)

    # Loss config: little uses configured loss (matches A's proxy);
    # Independent baseline stays on plain CE for both roles by default.
    co = config.get("co_training", {})
    loss_type = co.get("little_loss_type", "ce") if role == "little" else "ce"
    label_smoothing = co.get("label_smoothing", 0.0)
    focal_gamma = co.get("focal_gamma", 2.0)
    seed = config.get("seed")

    log_path = experiment_dir / f"{role}_search_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_id", "generation", "accuracy", "params", "time_s", "genotype_str"])

    problem = SingleModelNASProblem(
        role=role,
        channel_options=channel_options,
        blocks_min=blocks_min,
        blocks_max=blocks_max,
        param_budget=sub_budget,
        min_param_budget=sub_min_budget,
        train_loader=train_loader,
        val_loader=val_loader,
        proxy_epochs=proxy_epochs,
        proxy_lr=proxy_lr,
        device=device,
        log_path=log_path,
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FeasibleSingleModelSampling(
            role=role,
            channel_options=channel_options,
            blocks_min=blocks_min,
            blocks_max=blocks_max,
            min_param_budget=sub_min_budget,
            param_budget=sub_budget,
            max_attempts_factor=config["search"].get("feasible_sampling_attempts_factor", 500),
            seed=seed,
        ),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        repair=FeasibleSingleModelRepair(
            role=role,
            channel_options=channel_options,
            blocks_min=blocks_min,
            blocks_max=blocks_max,
            min_param_budget=sub_min_budget,
            param_budget=sub_budget,
            max_attempts_factor=config["search"].get("feasible_sampling_attempts_factor", 500),
            seed=None if seed is None else seed + (1 if role == "little" else 2),
        ),
        eliminate_duplicates=True,
    )

    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)

    # Extract Pareto front results
    pareto_results: list[dict] = []
    for x, obj in zip(result.X, result.F):
        blocks = decode_single_vector(
            x.tolist(), role, channel_options, blocks_min, blocks_max,
        )
        pareto_results.append({
            "accuracy": -obj[0],
            "params": int(obj[1]),
            "blocks": blocks,
        })

    # Save Pareto front summary
    pareto_path = experiment_dir / f"{role}_pareto_front.csv"
    with open(pareto_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accuracy", "params", "n_blocks", "genotype"])
        for r in pareto_results:
            genotype_str = str([
                {"ch": b.channels, "ly": b.layers, "ks": b.kernel_size,
                 "ct": b.conv_type, "res": b.use_residual}
                for b in r["blocks"]
            ])
            writer.writerow([
                f"{r['accuracy']:.4f}", r["params"],
                len(r["blocks"]), genotype_str,
            ])

    print(f"\n[{role}] Search complete. {problem.eval_count} evaluations.")
    print(f"[{role}] Pareto front: {len(pareto_results)} solutions")
    print(f"[{role}] Results saved to: {experiment_dir}")

    return log_path, pareto_results


# ---------------------------------------------------------------------------
# 6. Combine Pareto fronts from independent searches
# ---------------------------------------------------------------------------

def combine_pareto_fronts(
    little_pareto: list[dict],
    big_pareto: list[dict],
    total_budget: int = 100_000,
    top_k: int = 5,
    threshold: float = 0.70,
    little_min_budget: int | None = None,
    little_budget: int | None = None,
    big_min_budget: int | None = None,
    big_budget: int | None = None,
    budget_tolerance: float = 0.35,
    require_big_more_blocks: bool = True,
    require_big_at_least_little_params: bool = True,
    min_total_budget: int | None = 50_000,
    max_memory_bytes: int | None = MEMORY_BUDGET_BYTES,
    return_metadata: bool = False,
) -> list[PairGenotype] | tuple[list[PairGenotype], list[dict]]:
    """Combine independently-searched Pareto fronts into feasible CascadePair genotypes.

    If per-role min/max budgets are provided, each front is first filtered to
    its role range. The combiner then enumerates every little-Pareto x
    big-Pareto pair, applies pair-level constraints, ranks only feasible pairs,
    and selects the final top_k cascades. This avoids dropping structurally
    compatible models too early, e.g. when the highest-accuracy big candidates
    are too large to pair with the highest-accuracy little candidates.
    """
    little_candidates = _filter_role_budget_candidates(
        little_pareto, little_budget, min_budget=little_min_budget,
    )
    big_candidates = _filter_role_budget_candidates(
        big_pareto, big_budget, min_budget=big_min_budget,
    )

    feasible_pairs: list[tuple[tuple[float, float, float], PairGenotype, dict]] = []
    for lm in little_candidates:
        for bm in big_candidates:
            if require_big_more_blocks and len(bm["blocks"]) <= len(lm["blocks"]):
                continue
            if require_big_at_least_little_params and bm["params"] < lm["params"]:
                continue
            total_params = lm["params"] + bm["params"]
            if min_total_budget is not None and total_params < min_total_budget:
                continue
            if total_params > total_budget:
                continue

            genotype = PairGenotype(
                little_blocks=lm["blocks"],
                big_blocks=bm["blocks"],
                threshold=threshold,
            )
            if max_memory_bytes is not None:
                memory = estimate_pair_memory(genotype)
                if memory["total_bytes"] > max_memory_bytes:
                    continue
            else:
                memory = estimate_pair_memory(genotype)

            # Prefer accurate standalone components, then pairs that use more
            # of the allowed budget without exceeding it.
            score = (
                lm["accuracy"] + bm["accuracy"],
                -abs(total_budget - total_params),
                -total_params,
            )
            metadata = {
                "little_proxy_acc": f"{lm['accuracy']:.4f}",
                "big_proxy_acc": f"{bm['accuracy']:.4f}",
                "little_search_params": lm["params"],
                "big_search_params": bm["params"],
                "little_search_blocks": len(lm["blocks"]),
                "big_search_blocks": len(bm["blocks"]),
                "selection_score": "|".join(str(x) for x in score),
            }
            feasible_pairs.append((score, genotype, metadata))

    feasible_pairs.sort(key=lambda item: item[0], reverse=True)
    selected = feasible_pairs[:top_k]
    genotypes = [genotype for _, genotype, _ in selected]
    metadata = [metadata for _, _, metadata in selected]
    if return_metadata:
        return genotypes, metadata
    return genotypes


def _filter_role_budget_candidates(
    pareto_front: list[dict],
    budget: int | None,
    min_budget: int | None = None,
) -> list[dict]:
    """Return all role candidates inside configured min/max parameter bounds."""
    feasible = pareto_front
    if budget is not None:
        feasible = [r for r in feasible if r["params"] <= budget]
    if min_budget is not None:
        feasible = [r for r in feasible if r["params"] >= min_budget]
    return feasible


def _select_near_budget_candidates(
    pareto_front: list[dict],
    budget: int | None,
    top_k: int,
    budget_tolerance: float,
    min_budget: int | None = None,
) -> list[dict]:
    """Pick high-accuracy Pareto candidates close to a role budget.

    The front may contain tiny diagnostic models as well as near-limit models.
    For role-aware independent cascades, hard-filter by min/max budget when
    configured, then prefer candidates in the upper budget band. If the search
    found no candidates in that band, fall back to the best feasible candidates
    inside the configured range.
    """
    if budget is None:
        feasible = pareto_front
        if min_budget is not None:
            feasible = [r for r in feasible if r["params"] >= min_budget]
        return sorted(feasible, key=lambda r: r["accuracy"], reverse=True)[:top_k]

    feasible = [r for r in pareto_front if r["params"] <= budget]
    if min_budget is not None:
        feasible = [r for r in feasible if r["params"] >= min_budget]
    if not feasible:
        return []

    lower_bound = max(min_budget or 0, budget * max(0.0, 1.0 - budget_tolerance))
    near_budget = [r for r in feasible if r["params"] >= lower_bound]
    candidates = near_budget if near_budget else feasible

    return sorted(
        candidates,
        key=lambda r: (r["accuracy"], -abs(budget - r["params"])),
        reverse=True,
    )[:top_k]


# ---------------------------------------------------------------------------
# 7. Evaluate combined cascades
# ---------------------------------------------------------------------------

def evaluate_combined_cascades(
    genotypes: list[PairGenotype],
    train_loader,
    eval_loader,
    thresholds: list[float],
    train_epochs: int,
    proxy_lr: float,
    device: torch.device,
    experiment_dir: Path,
    little_loss_type: str = "ce",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    results_filename: str = "combined_cascade_results.csv",
    genotype_metadata: list[dict] | None = None,
    save_checkpoints: bool = True,
) -> Path:
    """Train and evaluate combined cascade genotypes across thresholds.

    For each genotype, trains little and big models independently with CE,
    then evaluates cascade accuracy at each threshold on the supplied final
    evaluation loader. For CIFAR-10 runs, the caller passes the test loader so
    RQ1 plots compare independent and joint NAS on the same split.

    Records the same Meeting-4 metric set as the joint NAS path so RQ1 plots
    can compare on the same axes (avg cascade MACs vs cascade accuracy).
    """
    results_path = experiment_dir / results_filename
    metadata_columns: list[str] = []
    if genotype_metadata is not None:
        metadata_columns = sorted({
            key for row in genotype_metadata for key in row.keys()
        })

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(metadata_columns + [
            "genotype_id", "threshold",
            "cascade_acc", "exit_ratio",
            "little_acc", "big_acc",
            "little_params", "big_params", "total_params", "total_bytes",
            "little_flops", "big_flops", "cascade_flops",
            "little_ece", "routing_error_rate",
        ])

    for gid, genotype in enumerate(genotypes):
        print(f"\nEvaluating combined cascade {gid + 1}/{len(genotypes)}")

        # Train little model independently; mirror joint NAS little-path loss.
        import torch.nn.functional as F

        from src.training.trainer import focal_loss

        little_model = build_single_model(genotype.little_blocks).to(device)
        little_opt = torch.optim.SGD(
            little_model.parameters(), lr=proxy_lr, momentum=0.9, weight_decay=5e-4,
        )
        for _ in range(train_epochs):
            little_model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                little_opt.zero_grad()
                logits = little_model(inputs)
                if little_loss_type == "focal":
                    loss = focal_loss(
                        logits, targets,
                        gamma=focal_gamma, label_smoothing=label_smoothing,
                    )
                else:
                    loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
                loss.backward()
                little_opt.step()
        if save_checkpoints:
            torch.save(
                little_model.state_dict(),
                experiment_dir / f"indep_{gid}_little_final.pt",
            )

        # Train big model independently with the same plain CE setup.
        big_model = build_single_model(genotype.big_blocks).to(device)
        big_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        big_opt = torch.optim.SGD(
            big_model.parameters(), lr=proxy_lr, momentum=0.9, weight_decay=5e-4,
        )
        for _ in range(train_epochs):
            train_one_epoch(big_model, train_loader, big_criterion, big_opt, device)
        if save_checkpoints:
            torch.save(
                big_model.state_dict(),
                experiment_dir / f"indep_{gid}_big_final.pt",
            )

        little_params = count_parameters(little_model)
        big_params = count_parameters(big_model)
        total_params = little_params + big_params

        # Analytical MACs and memory bytes (consistent with joint NAS reporting)
        little_flops = _estimate_path_flops(genotype.little_blocks, INPUT_CHANNELS)
        big_flops = _estimate_path_flops(genotype.big_blocks, INPUT_CHANNELS)
        l_w, l_p = _estimate_path_memory(genotype.little_blocks, INPUT_CHANNELS)
        b_w, b_p = _estimate_path_memory(genotype.big_blocks, INPUT_CHANNELS)
        input_bytes = 32 * 32 * INPUT_CHANNELS * BYTES_PER_PARAM
        total_bytes = l_w + b_w + max(l_p, b_p) + input_bytes

        # Evaluate standalone accuracies (criterion is reporting-only).
        eval_criterion = nn.CrossEntropyLoss()
        _, little_acc = evaluate(little_model, eval_loader, eval_criterion, device)
        _, big_acc = evaluate(big_model, eval_loader, eval_criterion, device)

        # Per-threshold cascade evaluation with the full metric set.
        for threshold in thresholds:
            metrics = _evaluate_cascade_independent(
                little_model, big_model, eval_loader, device, threshold,
            )
            cflops = cascade_flops(little_flops, big_flops, metrics["exit_ratio"])
            print(
                f"  Genotype {gid} | Thr {threshold:.2f} | "
                f"Cascade {metrics['cascade_acc']:.4f} | Exit {metrics['exit_ratio']:.2f} | "
                f"Little {little_acc:.4f} | Big {big_acc:.4f} | "
                f"MACs {cflops:>10,.0f} | Bytes {total_bytes:>7,} | "
                f"ECE {metrics['little_ece']:.4f} | RoutErr {metrics['routing_error_rate']:.4f}"
            )
            with open(results_path, "a", newline="") as f:
                writer = csv.writer(f)
                metadata_values: list[str] = []
                if genotype_metadata is not None:
                    metadata = genotype_metadata[gid]
                    metadata_values = [str(metadata.get(col, "")) for col in metadata_columns]

                writer.writerow(metadata_values + [
                    gid, f"{threshold:.2f}",
                    f"{metrics['cascade_acc']:.4f}", f"{metrics['exit_ratio']:.4f}",
                    f"{little_acc:.4f}", f"{big_acc:.4f}",
                    little_params, big_params, total_params, total_bytes,
                    little_flops, big_flops, f"{cflops:.0f}",
                    f"{metrics['little_ece']:.4f}",
                    f"{metrics['routing_error_rate']:.4f}",
                ])

    print(f"\nCombined cascade evaluation saved to: {results_path}")
    return results_path


@torch.no_grad()
def _evaluate_cascade_independent(
    little_model: nn.Module,
    big_model: nn.Module,
    loader,
    device: torch.device,
    threshold: float,
) -> dict:
    """Evaluate cascade with two independently-trained models.

    Returns dict with: cascade_acc, exit_ratio, little_ece, routing_error_rate.
    """
    little_model.eval()
    big_model.eval()
    correct, total, exit_count, routing_err_count = 0, 0, 0, 0
    all_conf, all_corr = [], []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        little_out = little_model(inputs)
        big_out = big_model(inputs)

        confidence = torch.softmax(little_out, dim=1).max(dim=1).values
        little_pred = little_out.argmax(dim=1)
        big_pred = big_out.argmax(dim=1)
        use_little = confidence > threshold

        little_correct = little_pred.eq(targets)
        big_correct = big_pred.eq(targets)

        pred = torch.where(use_little, little_pred, big_pred)
        correct += pred.eq(targets).sum().item()
        exit_count += use_little.sum().item()
        routing_err_count += (use_little & ~little_correct & big_correct).sum().item()
        total += inputs.size(0)

        all_conf.append(confidence.cpu())
        all_corr.append(little_correct.cpu())

    conf_np = torch.cat(all_conf).numpy()
    corr_np = torch.cat(all_corr).numpy().astype(float)
    little_ece = compute_ece(conf_np, corr_np)

    return {
        "cascade_acc": correct / total,
        "exit_ratio": exit_count / total,
        "little_ece": little_ece,
        "routing_error_rate": routing_err_count / total,
    }
