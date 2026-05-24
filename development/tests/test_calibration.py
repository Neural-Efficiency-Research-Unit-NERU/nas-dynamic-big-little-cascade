"""Tests for ECE calibration metric and temperature scaling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.training.utils import compute_ece


def test_ece_perfect_calibration():
    """When confidence exactly matches accuracy per bin, ECE should be 0."""
    # All at conf=1.0 and all correct => ECE=0
    confidences = np.ones(100)
    accuracies = np.ones(100)
    ece = compute_ece(confidences, accuracies)
    assert abs(ece) < 1e-9


def test_ece_all_correct_high_confidence():
    """All predictions correct at 100% confidence => ECE near 0."""
    confidences = np.ones(50)
    accuracies = np.ones(50)
    ece = compute_ece(confidences, accuracies)
    assert ece < 1e-9


def test_ece_all_wrong_high_confidence():
    """All predictions wrong at 100% confidence => ECE near 1.0."""
    confidences = np.ones(50)
    accuracies = np.zeros(50)
    ece = compute_ece(confidences, accuracies)
    assert abs(ece - 1.0) < 1e-9


def test_ece_known_value():
    """Hand-computed ECE for controlled input."""
    # 100 samples: 50 at conf=0.9 (40 correct, 10 wrong => acc=0.8)
    # 50 at conf=0.5 (30 correct, 20 wrong => acc=0.6)
    confidences = np.array([0.9] * 50 + [0.5] * 50)
    accuracies = np.array([1.0] * 40 + [0.0] * 10 + [1.0] * 30 + [0.0] * 20)

    # Bin 0.9: |0.8 - 0.9| = 0.1, weight = 0.5
    # Bin 0.5: |0.6 - 0.5| = 0.1, weight = 0.5
    # ECE = 0.5*0.1 + 0.5*0.1 = 0.1
    ece = compute_ece(confidences, accuracies)
    assert abs(ece - 0.1) < 1e-9


def test_ece_empty_input():
    """Empty arrays should return 0."""
    ece = compute_ece(np.array([]), np.array([]))
    assert ece == 0.0


def test_ece_single_sample():
    """Single sample should return |accuracy - confidence|."""
    ece = compute_ece(np.array([0.7]), np.array([1.0]))
    assert abs(ece - 0.3) < 1e-9


# --- Temperature scaling tests ---


def test_temperature_scaling_reduces_confidence():
    """Temperature > 1 should reduce max softmax confidence."""
    logits = torch.tensor([[2.0, 0.5, 0.1]])
    conf_t1 = torch.softmax(logits / 1.0, dim=1).max().item()
    conf_t2 = torch.softmax(logits / 2.0, dim=1).max().item()
    assert conf_t2 < conf_t1


def test_temperature_scaling_preserves_predictions():
    """Temperature scaling should not change argmax predictions."""
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 3.0, 0.5]])
    pred_t1 = torch.softmax(logits / 1.0, dim=1).argmax(dim=1)
    pred_t3 = torch.softmax(logits / 3.0, dim=1).argmax(dim=1)
    assert torch.equal(pred_t1, pred_t3)


def test_temperature_one_is_identity():
    """Temperature = 1.0 should not change softmax output."""
    logits = torch.tensor([[1.5, -0.5, 0.3]])
    probs_raw = torch.softmax(logits, dim=1)
    probs_t1 = torch.softmax(logits / 1.0, dim=1)
    assert torch.allclose(probs_raw, probs_t1)


# --- Focal loss tests ---

def test_focal_loss_matches_ce_at_gamma_zero():
    """When gamma=0, focal loss should equal standard cross-entropy."""
    import torch.nn.functional as F
    from src.training.trainer import focal_loss
    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    fl = focal_loss(logits, targets, gamma=0.0)
    ce = F.cross_entropy(logits, targets)
    assert torch.allclose(fl, ce, atol=1e-5)


def test_focal_loss_lower_for_confident():
    """Focal loss should be lower than CE for well-classified batches."""
    import torch.nn.functional as F
    from src.training.trainer import focal_loss
    logits = torch.zeros(32, 10)
    targets = torch.arange(32) % 10
    logits[range(32), targets] = 5.0
    fl = focal_loss(logits, targets, gamma=2.0)
    ce = F.cross_entropy(logits, targets)
    assert fl < ce


def test_focal_loss_positive():
    """Focal loss should always be non-negative."""
    from src.training.trainer import focal_loss
    logits = torch.randn(64, 10)
    targets = torch.randint(0, 10, (64,))
    fl = focal_loss(logits, targets, gamma=2.0)
    assert fl.item() >= 0.0


def test_focal_loss_with_label_smoothing():
    """Focal loss should accept label_smoothing parameter."""
    from src.training.trainer import focal_loss
    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    fl_smooth = focal_loss(logits, targets, gamma=2.0, label_smoothing=0.1)
    fl_no_smooth = focal_loss(logits, targets, gamma=2.0, label_smoothing=0.0)
    assert fl_smooth.item() >= 0.0
    assert fl_no_smooth.item() >= 0.0


def test_focal_loss_gradient_flows():
    """Focal loss should produce gradients for backpropagation."""
    from src.training.trainer import focal_loss
    logits = torch.randn(16, 10, requires_grad=True)
    targets = torch.randint(0, 10, (16,))
    fl = focal_loss(logits, targets, gamma=2.0)
    fl.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_cascade_aware_joint_loss_is_scalar_and_backprops():
    """Cascade-aware loss should train both branches from one scalar."""
    from src.training.trainer import cascade_aware_joint_loss

    little_logits = torch.randn(8, 10, requires_grad=True)
    big_logits = torch.randn(8, 10, requires_grad=True)
    targets = torch.randint(0, 10, (8,))

    loss = cascade_aware_joint_loss(
        little_logits, big_logits, targets,
        threshold=0.8,
        label_smoothing=0.1,
        little_loss_type="ce",
    )
    loss.backward()

    assert loss.ndim == 0
    assert little_logits.grad is not None
    assert big_logits.grad is not None
    assert torch.isfinite(little_logits.grad).all()
    assert torch.isfinite(big_logits.grad).all()


def test_cascade_aware_joint_loss_changes_with_threshold():
    """The searched threshold should affect the proxy-training objective."""
    from src.training.trainer import cascade_aware_joint_loss

    little_logits = torch.tensor([
        [4.0, 0.1, 0.0],
        [0.2, 2.0, 0.1],
        [0.2, 0.1, 1.8],
    ])
    big_logits = torch.tensor([
        [2.0, 0.3, 0.1],
        [0.1, 2.2, 0.4],
        [0.3, 0.2, 2.0],
    ])
    targets = torch.tensor([0, 1, 2])

    low_threshold_loss = cascade_aware_joint_loss(
        little_logits, big_logits, targets,
        threshold=0.5,
        defer_cost_weight=0.01,
    )
    high_threshold_loss = cascade_aware_joint_loss(
        little_logits, big_logits, targets,
        threshold=0.95,
        defer_cost_weight=0.01,
    )

    assert not torch.allclose(low_threshold_loss, high_threshold_loss)
