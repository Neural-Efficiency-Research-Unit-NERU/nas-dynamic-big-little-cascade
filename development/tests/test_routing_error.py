"""Tests for the cascade routing_error_rate metric.

routing_error_rate is the fraction of samples on which the little
model exits with a wrong prediction while the big model would have been right.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.utils import evaluate_at_thresholds


class _DummyPair(torch.nn.Module):
    """Returns canned (little_logits, big_logits) per sample.

    Little always predicts class 0 with a tunable confidence. Big is an oracle
    (predicts the true label). So every class-1 sample on which little exits
    is "little exits wrong, big would be right" - the routing_error_rate
    definition. The confidence is tuned via `little_logit` so we can test both
    "little exits" and "little does not exit" regimes deterministically.
    """

    def __init__(self, n_classes: int = 2, little_logit: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.little_logit = little_logit  # logit on class 0; class 1 gets 0.0

    def forward(self, x: torch.Tensor):
        b = x.size(0)
        little = torch.zeros((b, self.n_classes))
        little[:, 0] = self.little_logit
        # big: oracle - predicts the label encoded in the input
        big = torch.full((b, self.n_classes), 0.0)
        labels = x[:, 0].long()
        for i in range(b):
            big[i, labels[i]] = 10.0
        return little, big


def _make_loader(targets: list[int]) -> DataLoader:
    # The "input" is a single channel storing the target label, so the dummy big
    # model can act as an oracle. Shape (N, 1) is enough.
    x = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(targets, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_routing_error_matches_class1_fraction():
    """All class-1 samples must contribute to routing_error at low thresholds.

    Little has logit (1.0, 0.0) -> softmax confidence on class 0 is ~0.731.
    With threshold 0.5 little always exits. Big is an oracle. Therefore
    routing_error_rate == fraction of class-1 samples.
    """
    targets = [0, 0, 1, 1, 1, 0]  # 3/6 are class 1
    loader = _make_loader(targets)
    model = _DummyPair(n_classes=2, little_logit=1.0)
    device = torch.device("cpu")
    out = evaluate_at_thresholds(
        model, loader, device, thresholds=[0.5], temperature=1.0,
    )
    assert len(out) == 1
    expected = sum(1 for t in targets if t == 1) / len(targets)
    assert abs(out[0]["routing_error_rate"] - expected) < 1e-6
    # Sanity: with little exiting on all 6 samples and being wrong on 3,
    # cascade accuracy under exit-only routing should be 3/6.
    assert abs(out[0]["cascade_acc"] - 0.5) < 1e-6
    assert abs(out[0]["exit_ratio"] - 1.0) < 1e-6


def test_routing_error_zero_when_threshold_above_confidence():
    """If threshold > little's confidence, little never exits.

    Little logit (1.0, 0.0) -> max softmax ~0.731. Threshold 0.8 blocks all
    exits, so all samples go to big (oracle). routing_error_rate must be 0
    because little never exits at all.
    """
    targets = [0, 0, 1, 1, 1, 0]
    loader = _make_loader(targets)
    model = _DummyPair(n_classes=2, little_logit=1.0)
    device = torch.device("cpu")
    out = evaluate_at_thresholds(
        model, loader, device, thresholds=[0.8], temperature=1.0,
    )
    assert out[0]["routing_error_rate"] == 0.0
    assert out[0]["exit_ratio"] == 0.0
    # Big is oracle -> cascade should be perfect.
    assert abs(out[0]["cascade_acc"] - 1.0) < 1e-6
