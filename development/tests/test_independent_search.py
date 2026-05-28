"""Tests for independent NAS baseline module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

import numpy as np
import torch
import pytest
from src.nas.search_space import (
    BlockGene, PairGenotype, LITTLE_CHANNELS, BIG_CHANNELS,
    LITTLE_BLOCKS_MIN, LITTLE_BLOCKS_MAX, BIG_BLOCKS_MIN, BIG_BLOCKS_MAX,
    INPUT_CHANNELS, _GENES_PER_BLOCK,
)
from src.nas.independent_search import (
    build_single_model, decode_single_vector, encode_single_blocks,
    combine_pareto_fronts, _evaluate_cascade_independent,
    _select_near_budget_candidates, FeasibleSingleModelSampling,
    FeasibleSingleModelRepair, _single_model_bounds,
)
from src.training.utils import BYTES_PER_PARAM, _estimate_path_memory, count_parameters
from torch.utils.data import DataLoader, TensorDataset


def test_build_single_model_forward_shape():
    """Build from 2 little BlockGenes and verify output shape (4, 10)."""
    blocks = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    model = build_single_model(blocks, num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"


def test_build_single_model_big_channels():
    """Build from 4 big BlockGenes with BIG_CHANNELS values and verify forward."""
    blocks = [
        BlockGene(channels=BIG_CHANNELS[0], layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=BIG_CHANNELS[1], layers=2, kernel_size=3,
                  conv_type="depthwise_separable", use_residual=True),
        BlockGene(channels=BIG_CHANNELS[2], layers=1, kernel_size=5,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=BIG_CHANNELS[3], layers=2, kernel_size=5,
                  conv_type="depthwise_separable", use_residual=True),
    ]
    model = build_single_model(blocks, num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"


def test_single_vector_dimensionality():
    """Little n_var = LITTLE_BLOCKS_MAX * 5 + 1 = 16, big n_var = BIG_BLOCKS_MAX * 5 + 1 = 26."""
    little_n_var = LITTLE_BLOCKS_MAX * _GENES_PER_BLOCK + 1
    big_n_var = BIG_BLOCKS_MAX * _GENES_PER_BLOCK + 1
    assert little_n_var == 16, f"Expected 16, got {little_n_var}"
    assert big_n_var == 26, f"Expected 26, got {big_n_var}"


def test_single_vector_roundtrip_little():
    """Encode 2 little BlockGenes to vector, decode back. All genes should match."""
    blocks = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=12, layers=2, kernel_size=5,
                  conv_type="depthwise_separable", use_residual=True),
    ]
    vec = encode_single_blocks(blocks, LITTLE_CHANNELS, LITTLE_BLOCKS_MAX)
    assert len(vec) == LITTLE_BLOCKS_MAX * _GENES_PER_BLOCK + 1, (
        f"Expected vector length {LITTLE_BLOCKS_MAX * _GENES_PER_BLOCK + 1}, got {len(vec)}"
    )

    decoded = decode_single_vector(vec, "little", LITTLE_CHANNELS,
                                   LITTLE_BLOCKS_MIN, LITTLE_BLOCKS_MAX)
    assert len(decoded) == len(blocks)
    for b_orig, b_dec in zip(blocks, decoded):
        assert b_orig.channels == b_dec.channels
        assert b_orig.layers == b_dec.layers
        assert b_orig.kernel_size == b_dec.kernel_size
        assert b_orig.conv_type == b_dec.conv_type
        assert b_orig.use_residual == b_dec.use_residual


def test_single_vector_roundtrip_big():
    """Encode 4 big BlockGenes, decode back. All genes should match."""
    blocks = [
        BlockGene(channels=16, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=24, layers=2, kernel_size=5,
                  conv_type="depthwise_separable", use_residual=True),
        BlockGene(channels=32, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=True),
        BlockGene(channels=48, layers=2, kernel_size=5,
                  conv_type="depthwise_separable", use_residual=False),
    ]
    vec = encode_single_blocks(blocks, BIG_CHANNELS, BIG_BLOCKS_MAX)
    assert len(vec) == BIG_BLOCKS_MAX * _GENES_PER_BLOCK + 1, (
        f"Expected vector length {BIG_BLOCKS_MAX * _GENES_PER_BLOCK + 1}, got {len(vec)}"
    )

    decoded = decode_single_vector(vec, "big", BIG_CHANNELS,
                                   BIG_BLOCKS_MIN, BIG_BLOCKS_MAX)
    assert len(decoded) == len(blocks)
    for b_orig, b_dec in zip(blocks, decoded):
        assert b_orig.channels == b_dec.channels
        assert b_orig.layers == b_dec.layers
        assert b_orig.kernel_size == b_dec.kernel_size
        assert b_orig.conv_type == b_dec.conv_type
        assert b_orig.use_residual == b_dec.use_residual


def test_single_model_param_count_reasonable():
    """Random little models should remain inside the embedded-search scale."""
    random.seed(42)
    for _ in range(20):
        n_blocks = random.randint(LITTLE_BLOCKS_MIN, LITTLE_BLOCKS_MAX)
        blocks = [
            BlockGene(
                channels=random.choice(LITTLE_CHANNELS),
                layers=random.choice([1, 2]),
                kernel_size=random.choice([3, 5]),
                conv_type=random.choice(["standard", "depthwise_separable"]),
                use_residual=random.choice([True, False]),
            )
            for _ in range(n_blocks)
        ]
        model = build_single_model(blocks, num_classes=10)
        n_params = count_parameters(model)
        assert n_params < 100_000, (
            f"Model with {n_blocks} blocks has {n_params:,} params (expected < 100K)"
        )


def _make_single_dummy_problem(channel_options, blocks_min, blocks_max):
    """Build the minimal Problem-like object FloatRandomSampling needs."""
    xl_values, xu_values = _single_model_bounds(channel_options, blocks_min, blocks_max)

    class DummyProblem:
        def has_bounds(self):
            return True

        def bounds(self):
            return self.xl, self.xu

    problem = DummyProblem()
    problem.n_var = blocks_max * _GENES_PER_BLOCK + 1
    problem.xl = np.array(xl_values, dtype=float)
    problem.xu = np.array(xu_values, dtype=float)
    return problem


@pytest.mark.parametrize(
    "role,channel_options,blocks_min,blocks_max,min_params,max_params",
    [
        ("little", LITTLE_CHANNELS, LITTLE_BLOCKS_MIN, LITTLE_BLOCKS_MAX, 15_000, 30_000),
        ("big", BIG_CHANNELS, BIG_BLOCKS_MIN, BIG_BLOCKS_MAX, 40_000, 85_000),
    ],
)
def test_feasible_single_model_sampling_respects_role_budget(
    role, channel_options, blocks_min, blocks_max, min_params, max_params,
):
    """Independent initial sampling should spend population slots on trainable candidates."""
    problem = _make_single_dummy_problem(channel_options, blocks_min, blocks_max)
    sampler = FeasibleSingleModelSampling(
        role=role,
        channel_options=channel_options,
        blocks_min=blocks_min,
        blocks_max=blocks_max,
        min_param_budget=min_params,
        param_budget=max_params,
    )

    samples = sampler._do(problem, 5)

    assert samples.shape == (5, blocks_max * _GENES_PER_BLOCK + 1)
    for x in samples:
        blocks = decode_single_vector(
            x.tolist(), role, channel_options, blocks_min, blocks_max,
        )
        weight_bytes, _ = _estimate_path_memory(blocks, INPUT_CHANNELS)
        n_params = weight_bytes // BYTES_PER_PARAM
        assert min_params <= n_params <= max_params


def test_feasible_single_model_repair_replaces_infeasible_offspring():
    """Independent offspring outside the role budget should be repaired before evaluation."""
    problem = _make_single_dummy_problem(
        LITTLE_CHANNELS, LITTLE_BLOCKS_MIN, LITTLE_BLOCKS_MAX,
    )
    # One tiny, low-channel, one-block child is below the little 15K floor.
    infeasible = np.array([[0.0] * (LITTLE_BLOCKS_MAX * _GENES_PER_BLOCK) + [1.0]])
    repair = FeasibleSingleModelRepair(
        role="little",
        channel_options=LITTLE_CHANNELS,
        blocks_min=LITTLE_BLOCKS_MIN,
        blocks_max=LITTLE_BLOCKS_MAX,
        min_param_budget=15_000,
        param_budget=30_000,
    )

    repaired = repair._do(problem, infeasible)
    blocks = decode_single_vector(
        repaired[0].tolist(), "little", LITTLE_CHANNELS,
        LITTLE_BLOCKS_MIN, LITTLE_BLOCKS_MAX,
    )
    weight_bytes, _ = _estimate_path_memory(blocks, INPUT_CHANNELS)
    n_params = weight_bytes // BYTES_PER_PARAM

    assert 15_000 <= n_params <= 30_000


def _make_pareto_entry(accuracy: float, params: int,
                       blocks: list[BlockGene]) -> dict:
    """Helper to create a mock pareto front entry."""
    return {"accuracy": accuracy, "params": params, "blocks": blocks}


def test_combine_pareto_fronts_budget_filter():
    """Little (10K) + Big (95K) = 105K > 100K budget -> empty result."""
    little_blocks = [
        BlockGene(channels=4, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    big_blocks = [
        BlockGene(channels=32, layers=2, kernel_size=5,
                  conv_type="standard", use_residual=True),
        BlockGene(channels=32, layers=2, kernel_size=5,
                  conv_type="standard", use_residual=True),
    ]
    little_pareto = [_make_pareto_entry(0.70, 10_000, little_blocks)]
    big_pareto = [_make_pareto_entry(0.85, 95_000, big_blocks)]

    result = combine_pareto_fronts(little_pareto, big_pareto,
                                   total_budget=100_000, top_k=10)
    assert len(result) == 0, (
        f"Expected empty result for over-budget pair, got {len(result)}"
    )


def test_combine_pareto_fronts_within_budget():
    """Little (20K) + Big (60K) = 80K < 100K -> valid PairGenotype."""
    little_blocks = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    big_blocks = [
        BlockGene(channels=16, layers=2, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=24, layers=1, kernel_size=5,
                  conv_type="depthwise_separable", use_residual=True),
    ]
    little_pareto = [_make_pareto_entry(0.75, 20_000, little_blocks)]
    big_pareto = [_make_pareto_entry(0.88, 60_000, big_blocks)]

    result = combine_pareto_fronts(little_pareto, big_pareto,
                                   total_budget=100_000, top_k=10)
    assert len(result) >= 1, "Expected at least one valid pair"
    assert isinstance(result[0], PairGenotype)


def test_combine_pareto_fronts_uses_configured_threshold():
    """Independent cascade pairs should carry the configured base threshold."""
    little_blocks = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    big_blocks = [
        BlockGene(channels=16, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=24, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    little_pareto = [_make_pareto_entry(0.75, 20_000, little_blocks)]
    big_pareto = [_make_pareto_entry(0.88, 60_000, big_blocks)]

    result = combine_pareto_fronts(
        little_pareto, big_pareto, total_budget=100_000, top_k=10, threshold=0.70
    )

    assert result[0].threshold == 0.70


def test_combine_pareto_fronts_rejects_big_not_deeper_than_little():
    """Independent pairs require big to have more blocks than little."""
    little_blocks = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=12, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    big_blocks = [
        BlockGene(channels=16, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=24, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    little_pareto = [_make_pareto_entry(0.75, 20_000, little_blocks)]
    big_pareto = [_make_pareto_entry(0.88, 60_000, big_blocks)]

    result = combine_pareto_fronts(
        little_pareto, big_pareto, total_budget=100_000, top_k=10,
    )

    assert result == []


def test_combine_pareto_fronts_selects_pair_after_block_constraint():
    """Pair selection should not discard deeper big candidates too early."""
    little_blocks = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=12, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=16, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    big_same_depth_high_acc = [
        BlockGene(channels=16, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=24, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=32, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    big_deeper_feasible = big_same_depth_high_acc + [
        BlockGene(channels=48, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    little_pareto = [_make_pareto_entry(0.80, 24_000, little_blocks)]
    big_pareto = [
        _make_pareto_entry(0.95, 48_000, big_same_depth_high_acc),
        _make_pareto_entry(0.85, 43_000, big_deeper_feasible),
    ]

    result = combine_pareto_fronts(
        little_pareto,
        big_pareto,
        total_budget=100_000,
        top_k=1,
        little_min_budget=15_000,
        little_budget=30_000,
        big_min_budget=40_000,
        big_budget=85_000,
        require_big_more_blocks=True,
    )

    assert len(result) == 1
    assert len(result[0].big_blocks) == 4


def test_combine_pareto_fronts_enumerates_all_feasible_pairs():
    """Oversized top role candidates should not prevent feasible lower pairs."""
    little_30k = [
        BlockGene(channels=16, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=24, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    little_15k = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    big_85k = [
        BlockGene(channels=48, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=48, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=64, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    big_70k = [
        BlockGene(channels=32, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=48, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=48, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    little_pareto = [
        _make_pareto_entry(0.90, 30_000, little_30k),
        _make_pareto_entry(0.80, 15_000, little_15k),
    ]
    big_pareto = [
        _make_pareto_entry(0.95, 85_000, big_85k),
        _make_pareto_entry(0.85, 70_000, big_70k),
    ]

    result = combine_pareto_fronts(
        little_pareto,
        big_pareto,
        total_budget=100_000,
        top_k=3,
        little_min_budget=15_000,
        little_budget=30_000,
        big_min_budget=40_000,
        big_budget=85_000,
        require_big_more_blocks=True,
        max_memory_bytes=None,
    )

    # 30K+85K is invalid, but 30K+70K and 15K+85K are still considered.
    assert len(result) >= 2
    assert any(g.little_blocks == little_30k and g.big_blocks == big_70k for g in result)
    assert any(g.little_blocks == little_15k and g.big_blocks == big_85k for g in result)


def test_combine_pareto_fronts_rejects_memory_infeasible_pairs():
    """Independent pair generation should honor deployment-memory budget."""
    little_blocks = [
        BlockGene(channels=16, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    big_blocks = [
        BlockGene(channels=48, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=64, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    little_pareto = [_make_pareto_entry(0.80, 15_000, little_blocks)]
    big_pareto = [_make_pareto_entry(0.90, 40_000, big_blocks)]

    result = combine_pareto_fronts(
        little_pareto,
        big_pareto,
        total_budget=100_000,
        top_k=5,
        little_min_budget=15_000,
        little_budget=30_000,
        big_min_budget=40_000,
        big_budget=85_000,
        max_memory_bytes=1,
    )

    assert result == []


def test_combine_pareto_fronts_generates_valid_genotypes():
    """Returned PairGenotype objects have correct structure."""
    little_blocks = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=12, layers=2, kernel_size=5,
                  conv_type="depthwise_separable", use_residual=True),
    ]
    big_blocks = [
        BlockGene(channels=16, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=24, layers=2, kernel_size=5,
                  conv_type="depthwise_separable", use_residual=True),
        BlockGene(channels=32, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    little_pareto = [_make_pareto_entry(0.80, 15_000, little_blocks)]
    big_pareto = [_make_pareto_entry(0.90, 40_000, big_blocks)]

    result = combine_pareto_fronts(little_pareto, big_pareto,
                                   total_budget=100_000, top_k=10)
    assert len(result) >= 1
    for genotype in result:
        assert isinstance(genotype, PairGenotype)
        assert isinstance(genotype.little_blocks, list)
        assert isinstance(genotype.big_blocks, list)
        assert all(isinstance(b, BlockGene) for b in genotype.little_blocks)
        assert all(isinstance(b, BlockGene) for b in genotype.big_blocks)
        assert len(genotype.little_blocks) > 0
        assert len(genotype.big_blocks) > 0


def test_combine_pareto_fronts_empty_inputs():
    """Empty little pareto -> empty result. Empty big pareto -> empty result."""
    some_blocks = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    non_empty = [_make_pareto_entry(0.80, 15_000, some_blocks)]

    # Empty little
    result = combine_pareto_fronts([], non_empty,
                                   total_budget=100_000, top_k=10)
    assert len(result) == 0, "Expected empty result with empty little pareto"

    # Empty big
    result = combine_pareto_fronts(non_empty, [],
                                   total_budget=100_000, top_k=10)
    assert len(result) == 0, "Expected empty result with empty big pareto"

    # Both empty
    result = combine_pareto_fronts([], [], total_budget=100_000, top_k=10)
    assert len(result) == 0, "Expected empty result with both empty"


def test_evaluate_cascade_independent_returns_cascade_metrics():
    """_evaluate_cascade_independent must return the cascade metrics needed
    for joint-vs-independent comparison.

    Uses dummy little + big modules so the test stays fast (no training).
    """
    class _Little(torch.nn.Module):
        # Confidently predicts class 0 every time -> low confidence on threshold>0.7,
        # high confidence at threshold=0.5.
        def forward(self, x):
            b = x.size(0)
            out = torch.zeros((b, 2))
            out[:, 0] = 1.0
            return out

    class _BigOracle(torch.nn.Module):
        # Oracle: predicts the true label encoded in x[:, 0, 0, 0].long().
        def forward(self, x):
            b = x.size(0)
            labels = x[:, 0, 0, 0].long()
            out = torch.full((b, 2), 0.0)
            for i in range(b):
                out[i, labels[i]] = 10.0
            return out

    targets = [0, 0, 1, 1, 1, 0]
    # Build (3, 32, 32) inputs whose first pixel encodes the label.
    xs = torch.zeros((len(targets), 3, 32, 32))
    for i, t in enumerate(targets):
        xs[i, 0, 0, 0] = float(t)
    ys = torch.tensor(targets, dtype=torch.long)
    loader = DataLoader(TensorDataset(xs, ys), batch_size=4)

    metrics = _evaluate_cascade_independent(
        _Little(), _BigOracle(), loader, torch.device("cpu"), threshold=0.5,
    )

    # Must contain all four Meeting-4 keys.
    for key in ("cascade_acc", "exit_ratio", "little_ece", "routing_error_rate"):
        assert key in metrics, f"missing key: {key}"

    # All in [0, 1].
    for key in ("cascade_acc", "exit_ratio", "little_ece", "routing_error_rate"):
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


def test_combined_cascade_csv_header_includes_cascade_columns():
    """combined_cascade_results.csv must include the columns needed for
    joint-vs-independent comparison on accuracy, MACs, memory, and routing.

    We assert this against the actual CSV-writing code path by running
    `evaluate_combined_cascades` with a no-op training (proxy_epochs=0).
    """
    from pathlib import Path
    import csv as _csv
    import tempfile
    from src.nas.independent_search import evaluate_combined_cascades

    # Tiny dataset so training (epochs=0) is a no-op
    xs = torch.zeros((4, 3, 32, 32))
    ys = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(xs, ys), batch_size=2)

    little = [
        BlockGene(channels=4, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    big = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    g = PairGenotype(little_blocks=little, big_blocks=big, threshold=0.8)

    with tempfile.TemporaryDirectory() as td:
        out = evaluate_combined_cascades(
            [g], loader, loader, thresholds=[0.5, 0.95],
            train_epochs=0, proxy_lr=0.01,
            device=torch.device("cpu"), experiment_dir=Path(td),
        )
        with open(out) as f:
            header = next(_csv.reader(f))

        for col in ("cascade_flops", "total_bytes", "little_ece",
                    "routing_error_rate", "little_flops", "big_flops"):
            assert col in header, f"combined_cascade_results.csv missing {col}"


def test_combine_respects_top_k():
    """With top_k=2 and 5 entries each, at most 4 combinations (2x2)."""
    little_blocks_variants = [
        [BlockGene(channels=ch, layers=1, kernel_size=3,
                   conv_type="standard", use_residual=False)]
        for ch in [8, 12, 16, 24, 32]
    ]
    big_blocks_variants = [
        [BlockGene(channels=ch, layers=1, kernel_size=3,
                   conv_type="standard", use_residual=False),
         BlockGene(channels=ch, layers=1, kernel_size=3,
                   conv_type="standard", use_residual=False)]
        for ch in [16, 24, 32, 48, 64]
    ]

    little_pareto = [
        _make_pareto_entry(0.70 + i * 0.02, 5_000 + i * 1_000, blocks)
        for i, blocks in enumerate(little_blocks_variants)
    ]
    big_pareto = [
        _make_pareto_entry(0.80 + i * 0.02, 10_000 + i * 2_000, blocks)
        for i, blocks in enumerate(big_blocks_variants)
    ]

    result = combine_pareto_fronts(little_pareto, big_pareto,
                                   total_budget=1_000_000, top_k=2)
    # top_k=2 selects 2 from each front -> at most 2*2=4 combinations
    assert len(result) <= 4, (
        f"Expected at most 4 results with top_k=2, got {len(result)}"
    )


def test_select_near_budget_candidates_filters_to_role_budget():
    """Per-role selection must not let little consume the big role budget."""
    front = [
        _make_pareto_entry(0.90, 46_000, []),
        _make_pareto_entry(0.82, 28_000, []),
        _make_pareto_entry(0.70, 3_000, []),
    ]

    selected = _select_near_budget_candidates(
        front, budget=30_000, top_k=5, budget_tolerance=0.35,
    )

    assert all(r["params"] <= 30_000 for r in selected)
    assert selected[0]["params"] == 28_000


def test_select_near_budget_candidates_enforces_min_budget():
    """Per-role selection must reject tiny models when a lower bound is set."""
    front = [
        _make_pareto_entry(0.95, 5_000, []),
        _make_pareto_entry(0.82, 18_000, []),
        _make_pareto_entry(0.80, 28_000, []),
    ]

    selected = _select_near_budget_candidates(
        front, budget=30_000, top_k=5, budget_tolerance=0.35,
        min_budget=15_000,
    )

    assert all(15_000 <= r["params"] <= 30_000 for r in selected)
    assert all(r["params"] != 5_000 for r in selected)


def test_combine_pareto_fronts_uses_role_budgets():
    """Independent pair generation should honor little=30K and big=85K."""
    little_small = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    little_too_large = [
        BlockGene(channels=32, layers=2, kernel_size=5,
                  conv_type="standard", use_residual=False),
    ]
    big_near_limit = [
        BlockGene(channels=48, layers=2, kernel_size=5,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=48, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    little_pareto = [
        _make_pareto_entry(0.95, 46_000, little_too_large),
        _make_pareto_entry(0.80, 29_000, little_small),
    ]
    big_pareto = [
        _make_pareto_entry(0.88, 65_000, big_near_limit),
    ]

    result = combine_pareto_fronts(
        little_pareto,
        big_pareto,
        total_budget=100_000,
        top_k=5,
        little_budget=30_000,
        big_budget=85_000,
    )

    assert len(result) == 1
    assert result[0].little_blocks == little_small


def test_combine_pareto_fronts_uses_role_min_budgets():
    """Independent pairs should honor little/big lower bounds."""
    tiny_little = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    valid_little = [
        BlockGene(channels=16, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    tiny_big = [
        BlockGene(channels=16, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    valid_big = [
        BlockGene(channels=48, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
        BlockGene(channels=48, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    little_pareto = [
        _make_pareto_entry(0.95, 5_000, tiny_little),
        _make_pareto_entry(0.80, 20_000, valid_little),
    ]
    big_pareto = [
        _make_pareto_entry(0.95, 20_000, tiny_big),
        _make_pareto_entry(0.85, 50_000, valid_big),
    ]

    result = combine_pareto_fronts(
        little_pareto,
        big_pareto,
        total_budget=100_000,
        top_k=5,
        little_min_budget=15_000,
        little_budget=30_000,
        big_min_budget=40_000,
        big_budget=85_000,
    )

    assert len(result) == 1
    assert result[0].little_blocks == valid_little
    assert result[0].big_blocks == valid_big


def test_proxy_train_single_accepts_focal_loss():
    """`proxy_train_single` must train under focal+smoothing when configured."""
    import torch
    from src.nas.independent_search import proxy_train_single

    torch.manual_seed(0)
    blocks = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    out = proxy_train_single(
        blocks, loader, loader,
        epochs=1, lr=0.01,
        device=torch.device("cpu"),
        loss_type="focal",
        label_smoothing=0.1,
        focal_gamma=2.0,
    )
    assert "accuracy" in out and "n_params" in out
    assert 0.0 <= out["accuracy"] <= 1.0


def test_proxy_train_single_defaults_to_ce():
    """Backwards-compat: omitting the loss kwargs uses CE."""
    import torch
    from src.nas.independent_search import proxy_train_single

    torch.manual_seed(0)
    blocks = [
        BlockGene(channels=8, layers=1, kernel_size=3,
                  conv_type="standard", use_residual=False),
    ]
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    out = proxy_train_single(
        blocks, loader, loader,
        epochs=1, lr=0.01, device=torch.device("cpu"),
    )
    assert 0.0 <= out["accuracy"] <= 1.0
