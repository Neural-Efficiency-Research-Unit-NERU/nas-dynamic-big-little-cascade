"""Tests for two-model co-search search space (Depth Explorer)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.nas.search_space import (
    LITTLE_CHANNELS, BIG_CHANNELS, CONV_TYPE_OPTIONS, KERNEL_OPTIONS,
    LAYERS_OPTIONS, LITTLE_BLOCKS_MIN, LITTLE_BLOCKS_MAX,
    BIG_BLOCKS_MIN, BIG_BLOCKS_MAX, THRESHOLD_MIN, THRESHOLD_MAX,
    N_VAR, INPUT_CHANNELS, XL, XU,
    BlockGene, PairGenotype,
    random_pair_genotype, pair_genotype_to_vector, vector_to_pair_genotype,
)
from src.nas.evolutionary import FeasiblePairRepair, FeasiblePairSampling


def test_search_space_constants():
    assert LITTLE_CHANNELS == [8, 12, 16, 24, 32]
    assert BIG_CHANNELS == [16, 24, 32, 48, 64]
    assert LAYERS_OPTIONS == [1, 2]
    assert KERNEL_OPTIONS == [3, 5]
    assert CONV_TYPE_OPTIONS == ["standard", "depthwise_separable"]
    assert LITTLE_BLOCKS_MIN == 1
    assert LITTLE_BLOCKS_MAX == 3
    assert BIG_BLOCKS_MIN == 2
    assert BIG_BLOCKS_MAX == 5
    assert N_VAR == 43
    assert INPUT_CHANNELS == 3


def test_pair_genotype_block_counts():
    for _ in range(30):
        g = random_pair_genotype()
        assert LITTLE_BLOCKS_MIN <= len(g.little_blocks) <= LITTLE_BLOCKS_MAX
        assert BIG_BLOCKS_MIN <= len(g.big_blocks) <= BIG_BLOCKS_MAX


def test_roundtrip_encoding():
    for _ in range(30):
        g = random_pair_genotype()
        vec = pair_genotype_to_vector(g)
        assert len(vec) == N_VAR
        g2 = vector_to_pair_genotype(vec)
        assert len(g2.little_blocks) == len(g.little_blocks)
        assert len(g2.big_blocks) == len(g.big_blocks)
        for b1, b2 in zip(g.little_blocks, g2.little_blocks):
            assert b1.channels == b2.channels
            assert b1.layers == b2.layers
            assert b1.kernel_size == b2.kernel_size
            assert b1.conv_type == b2.conv_type
            assert b1.use_residual == b2.use_residual
        for b1, b2 in zip(g.big_blocks, g2.big_blocks):
            assert b1.channels == b2.channels
            assert b1.layers == b2.layers
            assert b1.kernel_size == b2.kernel_size
            assert b1.conv_type == b2.conv_type
            assert b1.use_residual == b2.use_residual
        assert abs(g2.threshold - g.threshold) < 1e-6


def test_pair_genotype_serialization():
    for _ in range(20):
        g = random_pair_genotype()
        d = g.to_dict()
        assert "little" in d
        assert "big" in d
        assert "threshold" in d
        assert len(d["little"]) == len(g.little_blocks)
        assert len(d["big"]) == len(g.big_blocks)
        assert d["threshold"] == g.threshold
        g2 = PairGenotype.from_dict(d)
        assert len(g2.little_blocks) == len(g.little_blocks)
        assert len(g2.big_blocks) == len(g.big_blocks)
        assert g2.threshold == g.threshold
        assert g2.little_blocks[0].conv_type == g.little_blocks[0].conv_type


def test_bounds_length():
    assert len(XL) == N_VAR
    assert len(XU) == N_VAR


def test_vector_bounds_valid():
    for _ in range(30):
        g = random_pair_genotype()
        vec = pair_genotype_to_vector(g)
        for i, (v, lo, hi) in enumerate(zip(vec, XL, XU)):
            assert lo <= v <= hi, f"Variable {i}: {v} not in [{lo}, {hi}]"


def test_block_gene_has_conv_type():
    gene = BlockGene(channels=16, layers=1, kernel_size=3, conv_type="standard", use_residual=False)
    assert gene.conv_type == "standard"


# --- New tests for Depth Explorer ---

def test_asymmetric_channels():
    """Little blocks use LITTLE_CHANNELS, big blocks use BIG_CHANNELS."""
    for _ in range(50):
        g = random_pair_genotype()
        for b in g.little_blocks:
            assert b.channels in LITTLE_CHANNELS, f"Little channel {b.channels} not in {LITTLE_CHANNELS}"
        for b in g.big_blocks:
            assert b.channels in BIG_CHANNELS, f"Big channel {b.channels} not in {BIG_CHANNELS}"


def test_variable_depth_roundtrip():
    """Encoding/decoding preserves active block count across the full range."""
    seen_little = set()
    seen_big = set()
    for _ in range(100):
        g = random_pair_genotype()
        vec = pair_genotype_to_vector(g)
        g2 = vector_to_pair_genotype(vec)
        assert len(g2.little_blocks) == len(g.little_blocks)
        assert len(g2.big_blocks) == len(g.big_blocks)
        seen_little.add(len(g.little_blocks))
        seen_big.add(len(g.big_blocks))
    # Over 100 random samples, we should see variety
    assert len(seen_little) >= 2, f"Only saw little block counts: {seen_little}"
    assert len(seen_big) >= 2, f"Only saw big block counts: {seen_big}"


def test_threshold_roundtrip():
    """Threshold gene round-trips through encoding/decoding."""
    for _ in range(30):
        g = random_pair_genotype()
        vec = pair_genotype_to_vector(g)
        g2 = vector_to_pair_genotype(vec)
        assert abs(g2.threshold - g.threshold) < 1e-6


def test_threshold_clamping():
    """Out-of-range threshold values are clamped."""
    g = random_pair_genotype()
    vec = pair_genotype_to_vector(g)

    vec[42] = 0.30  # below THRESHOLD_MIN
    g2 = vector_to_pair_genotype(vec)
    assert g2.threshold >= THRESHOLD_MIN

    vec[42] = 1.50  # above THRESHOLD_MAX
    g3 = vector_to_pair_genotype(vec)
    assert g3.threshold <= THRESHOLD_MAX


def test_threshold_override_for_fixed_threshold_search():
    """Search can keep threshold fixed while preserving 43-dim vectors."""
    g = random_pair_genotype()
    vec = pair_genotype_to_vector(g)
    vec[42] = THRESHOLD_MAX
    g2 = vector_to_pair_genotype(vec, threshold_override=0.70)
    assert abs(g2.threshold - 0.70) < 1e-6


def test_feasible_pair_sampling_respects_param_floor():
    """Initial NSGA sampling should return real candidates, not tiny infeasible ones."""
    class DummyProblem:
        n_var = N_VAR
        xl = np.array(XL, dtype=float)
        xu = np.array(XU, dtype=float)

        def has_bounds(self):
            return True

        def bounds(self):
            return self.xl, self.xu

    sampler = FeasiblePairSampling(
        memory_budget_bytes=460_800,
        min_params=50_000,
        max_params=100_000,
        little_min_params=15_000,
        little_max_params=30_000,
        big_min_params=40_000,
        big_max_params=85_000,
        require_big_more_blocks=True,
        require_big_at_least_little_params=True,
        fixed_threshold=0.80,
    )
    samples = sampler._do(DummyProblem(), 5)
    assert samples.shape == (5, N_VAR)
    for x in samples:
        genotype = vector_to_pair_genotype(x.tolist(), threshold_override=0.80)
        # Import locally to keep this test focused on the sampling contract.
        from src.training.utils import estimate_pair_memory

        params = estimate_pair_memory(genotype)["total_params"]
        assert 50_000 <= params <= 100_000
        memory = estimate_pair_memory(genotype)
        assert memory["little_params"] >= 15_000
        assert memory["little_params"] <= 30_000
        assert memory["big_params"] >= 40_000
        assert memory["big_params"] <= 85_000
        assert len(genotype.big_blocks) > len(genotype.little_blocks)
        assert memory["big_params"] >= memory["little_params"]


def test_feasible_pair_repair_replaces_infeasible_offspring():
    """Co-search offspring should be made feasible before proxy training."""
    class DummyProblem:
        n_var = N_VAR
        xl = np.array(XL, dtype=float)
        xu = np.array(XU, dtype=float)

        def has_bounds(self):
            return True

        def bounds(self):
            return self.xl, self.xu

    infeasible = np.zeros((1, N_VAR), dtype=float)
    repair = FeasiblePairRepair(
        memory_budget_bytes=460_800,
        min_params=50_000,
        max_params=100_000,
        little_min_params=15_000,
        little_max_params=30_000,
        big_min_params=40_000,
        big_max_params=85_000,
        require_big_more_blocks=True,
        require_big_at_least_little_params=True,
        fixed_threshold=0.70,
    )

    repaired = repair._do(DummyProblem(), infeasible)
    genotype = vector_to_pair_genotype(repaired[0].tolist(), threshold_override=0.70)
    from src.training.utils import estimate_pair_memory

    memory = estimate_pair_memory(genotype)
    assert 50_000 <= memory["total_params"] <= 100_000
    assert memory["total_bytes"] <= 460_800
    assert memory["little_params"] >= 15_000
    assert memory["little_params"] <= 30_000
    assert memory["big_params"] >= 40_000
    assert memory["big_params"] <= 85_000
    assert len(genotype.big_blocks) > len(genotype.little_blocks)
    assert memory["big_params"] >= memory["little_params"]


def test_active_count_clamping():
    """Out-of-range active block counts are clamped."""
    g = random_pair_genotype()
    vec = pair_genotype_to_vector(g)

    vec[40] = 0.0  # below LITTLE_BLOCKS_MIN
    g2 = vector_to_pair_genotype(vec)
    assert len(g2.little_blocks) >= LITTLE_BLOCKS_MIN

    vec[40] = 10.0  # above LITTLE_BLOCKS_MAX
    g3 = vector_to_pair_genotype(vec)
    assert len(g3.little_blocks) <= LITTLE_BLOCKS_MAX

    vec[41] = 0.0  # below BIG_BLOCKS_MIN
    g4 = vector_to_pair_genotype(vec)
    assert len(g4.big_blocks) >= BIG_BLOCKS_MIN

    vec[41] = 10.0  # above BIG_BLOCKS_MAX
    g5 = vector_to_pair_genotype(vec)
    assert len(g5.big_blocks) <= BIG_BLOCKS_MAX


def test_backward_compat_from_dict():
    """Old genotype dicts without 'threshold' should load with default 0.80."""
    old_dict = {
        "little": [
            {"channels": 8, "layers": 1, "kernel_size": 3,
             "conv_type": "standard", "use_residual": False},
            {"channels": 16, "layers": 2, "kernel_size": 5,
             "conv_type": "depthwise_separable", "use_residual": True},
        ],
        "big": [
            {"channels": 16, "layers": 1, "kernel_size": 3,
             "conv_type": "standard", "use_residual": False},
        ] * 4,
    }
    g = PairGenotype.from_dict(old_dict)
    assert g.threshold == 0.80
    assert len(g.little_blocks) == 2
    assert len(g.big_blocks) == 4
