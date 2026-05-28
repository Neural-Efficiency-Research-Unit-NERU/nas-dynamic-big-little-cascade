"""Tests for knee-and-neighbors selection used by retrain_best.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from scripts.retrain_best import (  # noqa: E402
    filter_budget_feasible,
    select_balanced_candidates,
    select_knee_neighbors,
    write_selected_architectures,
)


def _pt(
    acc: float,
    cflops: float,
    exit_ratio: float | None = None,
    params: int = 50_000,
    bytes_: int = 250_000,
    genotype: str | None = None,
) -> dict:
    return {
        "cascade_acc": acc,
        "cascade_flops": cflops,
        "exit_ratio": exit_ratio,
        # extra fields the real pareto rows carry; selector should ignore them
        "total_params": params,
        "total_bytes": bytes_,
        "genotype": genotype,
    }


def test_returns_three_for_dense_front():
    pareto = [
        _pt(0.60, 1e5),
        _pt(0.80, 5e5),
        _pt(0.88, 1e6),  # likely knee
        _pt(0.92, 2e6),
        _pt(0.93, 3e6),
    ]
    picks = select_knee_neighbors(pareto, k=3)
    assert len(picks) == 3


def test_picks_are_contiguous_in_flops():
    pareto = [
        _pt(0.60, 1e5),
        _pt(0.80, 5e5),
        _pt(0.88, 1e6),  # knee
        _pt(0.92, 2e6),
        _pt(0.93, 3e6),
    ]
    picks = select_knee_neighbors(pareto, k=3)
    flops = [p["cascade_flops"] for p in picks]
    assert flops == sorted(flops), "picks should be returned sorted by MACs"
    sorted_full = sorted([p["cascade_flops"] for p in pareto])
    start = sorted_full.index(flops[0])
    assert sorted_full[start:start + 3] == flops, (
        "picks should be a contiguous window in the MAC-sorted front"
    )


def test_knee_in_middle_includes_knee():
    """For a sharply concave front, picks must include the geometric knee."""
    # Steep rise then plateau - knee at 1e5 (the elbow point).
    pareto = [
        _pt(0.30, 1e4),
        _pt(0.85, 1e5),  # knee - max perpendicular distance from anchor line
        _pt(0.86, 5e5),
        _pt(0.87, 1e6),
        _pt(0.88, 2e6),
    ]
    picks = select_knee_neighbors(pareto, k=3)
    flops = sorted(p["cascade_flops"] for p in picks)
    assert 1e5 in flops, "knee point must be included in the picks"


def test_collinear_front_picks_first_window():
    """When the front is exactly collinear, all perps are zero; argmax returns 0,
    triggering the left-endpoint branch which slides the window right."""
    pareto = [
        _pt(1.0, 0.0),
        _pt(0.75, 1.0),
        _pt(0.50, 2.0),
        _pt(0.25, 3.0),
        _pt(0.0, 4.0),
    ]
    picks = select_knee_neighbors(pareto, k=3)
    flops = sorted(p["cascade_flops"] for p in picks)
    assert flops == [0.0, 1.0, 2.0], (
        f"expected first 3 by flops on collinear front, got {flops}"
    )


def test_degenerate_duplicate_front_returns_unique_candidate():
    """Identical candidates are deduplicated before knee selection."""
    pareto = [_pt(0.5, 1e5), _pt(0.5, 1e5), _pt(0.5, 1e5), _pt(0.5, 1e5)]
    picks = select_knee_neighbors(pareto, k=3)
    assert len(picks) == 1


def test_sparse_front_returns_all_available():
    pareto = [_pt(0.80, 5e5), _pt(0.88, 1e6)]
    picks = select_knee_neighbors(pareto, k=3)
    assert len(picks) == 2


def test_single_point_front_returns_single_point():
    pareto = [_pt(0.80, 5e5)]
    picks = select_knee_neighbors(pareto, k=3)
    assert len(picks) == 1


def test_empty_front_raises():
    with pytest.raises(ValueError):
        select_knee_neighbors([], k=3)


def test_derive_exit_ratio_consistency():
    """exit_ratio derivation matches the cascade_flops formula."""
    little, big = 100_000.0, 1_000_000.0
    for true_er in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cflops = little + (1.0 - true_er) * big
        derived = 1.0 - (cflops - little) / big
        assert abs(derived - true_er) < 1e-9


def test_budget_filter_enforces_params_and_bytes():
    candidates = [
        _pt(0.80, 1e5, 0.5, params=99_000, bytes_=450_000),
        _pt(0.81, 1e5, 0.5, params=101_000, bytes_=450_000),
        _pt(0.82, 1e5, 0.5, params=99_000, bytes_=470_000),
    ]
    filtered = filter_budget_feasible(candidates, max_params=100_000, max_bytes=460_800)
    assert filtered == [candidates[0]]


def test_budget_filter_enforces_min_params():
    candidates = [
        _pt(0.80, 1e5, 0.5, params=20_000, bytes_=200_000),
        _pt(0.81, 1e5, 0.5, params=35_000, bytes_=250_000),
        _pt(0.82, 1e5, 0.5, params=55_000, bytes_=300_000),
    ]
    filtered = filter_budget_feasible(
        candidates, min_params=50_000, max_params=100_000, max_bytes=460_800
    )
    assert filtered == [candidates[2]]


def test_balanced_selection_prefers_useful_exit_window():
    candidates = [
        _pt(0.90, 3e6, 0.01),
        _pt(0.86, 1e6, 0.35),
        _pt(0.85, 5e5, 0.45),
        _pt(0.84, 8e5, 0.70),
    ]
    picks, reason = select_balanced_candidates(candidates, k=3)
    assert "balanced" in reason
    assert all(0.2 <= p["exit_ratio"] <= 0.8 for p in picks)
    assert len(picks) == 3


def test_balanced_selection_falls_back_to_top_accuracy_when_all_degenerate():
    candidates = [
        _pt(0.65, 6e5, 0.00),
        _pt(0.68, 3e6, 0.01),
        _pt(0.66, 7e5, 0.00),
        _pt(0.67, 2e6, 0.02),
    ]
    picks, reason = select_balanced_candidates(candidates, k=3)
    assert "fell back" in reason
    assert [p["cascade_acc"] for p in picks] == [0.68, 0.67, 0.66]


def test_write_selected_architectures_csv(tmp_path):
    out = tmp_path / "selected_architectures.csv"
    picks = [_pt(0.68, 3e6, 0.01, params=66_585, bytes_=311_396)]
    write_selected_architectures(
        out,
        picks,
        selection_rule="balanced",
        candidate_source="search_log",
        reason="fallback",
    )
    text = out.read_text()
    assert "rank,selection_rule,candidate_source,reason" in text
    assert "balanced,search_log,fallback" in text
    assert "0.6800,0.0100,3000000" in text
