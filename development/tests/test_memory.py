"""Tests for pair memory estimation and budget utilisation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nas.search_space import BlockGene, PairGenotype
from src.training.utils import (
    MEMORY_BUDGET_BYTES,
    PARAM_BUDGET_REFERENCE,
    budget_utilisation,
    estimate_pair_memory,
)


def _make_pair(little_ch: list[int], big_ch: list[int]) -> PairGenotype:
    return PairGenotype(
        little_blocks=[
            BlockGene(ch, layers=1, kernel_size=3, conv_type="standard", use_residual=False)
            for ch in little_ch
        ],
        big_blocks=[
            BlockGene(ch, layers=1, kernel_size=3, conv_type="standard", use_residual=False)
            for ch in big_ch
        ],
    )


def test_returns_all_components():
    g = _make_pair([16, 16], [16, 16, 16, 16])
    mem = estimate_pair_memory(g)
    for key in ["little_weight_bytes", "big_weight_bytes",
                "little_peak_activation", "big_peak_activation",
                "input_bytes", "total_bytes",
                "little_params", "big_params", "total_params"]:
        assert key in mem, f"Missing key: {key}"


def test_total_uses_max_of_peaks():
    g = _make_pair([8, 8], [48, 48, 48, 48])
    mem = estimate_pair_memory(g)
    expected_total = (
        mem["little_weight_bytes"] + mem["big_weight_bytes"]
        + max(mem["little_peak_activation"], mem["big_peak_activation"])
        + mem["input_bytes"]
    )
    assert mem["total_bytes"] == expected_total


def test_more_channels_means_more_memory():
    g_small = _make_pair([8, 8], [8, 8, 8, 8])
    g_large = _make_pair([48, 48], [48, 48, 48, 48])
    assert estimate_pair_memory(g_large)["total_bytes"] > estimate_pair_memory(g_small)["total_bytes"]


def test_depthwise_separable_uses_less_weight_memory():
    g_std = PairGenotype(
        little_blocks=[BlockGene(32, 1, 3, "standard", False) for _ in range(2)],
        big_blocks=[BlockGene(32, 1, 3, "standard", False) for _ in range(4)],
    )
    g_dws = PairGenotype(
        little_blocks=[BlockGene(32, 1, 3, "depthwise_separable", False) for _ in range(2)],
        big_blocks=[BlockGene(32, 1, 3, "depthwise_separable", False) for _ in range(4)],
    )
    mem_std = estimate_pair_memory(g_std)
    mem_dws = estimate_pair_memory(g_dws)
    total_std = mem_std["little_weight_bytes"] + mem_std["big_weight_bytes"]
    total_dws = mem_dws["little_weight_bytes"] + mem_dws["big_weight_bytes"]
    assert total_dws < total_std


def test_param_budget_feasible():
    g = _make_pair([16, 24], [16, 24, 32, 32])
    mem = estimate_pair_memory(g)
    # Diagnostic reference still tracked, but no longer the constraint.
    assert mem["total_params"] <= PARAM_BUDGET_REFERENCE


def test_memory_budget_feasible():
    """Hard constraint: total deployment bytes <= MEMORY_BUDGET_BYTES."""
    g = _make_pair([16, 24], [16, 24, 32, 32])
    mem = estimate_pair_memory(g)
    assert mem["total_bytes"] <= MEMORY_BUDGET_BYTES


def test_budget_utilisation_in_unit_interval():
    g = _make_pair([16, 24], [16, 24, 32, 32])
    mem = estimate_pair_memory(g)
    util = budget_utilisation(mem["total_bytes"])
    assert 0.0 < util <= 1.0


def test_budget_utilisation_default_budget():
    """budget_utilisation defaults to MEMORY_BUDGET_BYTES."""
    half = MEMORY_BUDGET_BYTES // 2
    assert budget_utilisation(half) == 0.5
    assert budget_utilisation(MEMORY_BUDGET_BYTES) == 1.0


def test_total_params_consistency():
    g = _make_pair([16, 24], [16, 24, 32, 32])
    mem = estimate_pair_memory(g)
    assert mem["total_params"] == mem["little_params"] + mem["big_params"]
