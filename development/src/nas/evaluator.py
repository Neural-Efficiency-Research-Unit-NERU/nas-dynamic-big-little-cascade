"""Proxy training evaluator for co-searched CascadePair."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.paired_models import CascadePair
from src.nas.search_space import PairGenotype
from src.training.utils import compute_ece, count_parameters, estimate_pair_memory, estimate_pair_flops, cascade_flops
from src.training.trainer import branch_loss_per_sample, cascade_aware_joint_loss


def evaluate_cascade(
    model: nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.8,
) -> dict:
    """Evaluate cascade accuracy: little-first, big-fallback.

    Also tracks routing_error_rate: the fraction of samples where the little
    model exits with a wrong prediction while the big model would have been
    correct.
    """
    model.eval()
    correct, total = 0, 0
    little_correct, big_correct = 0, 0
    exit_count = 0
    routing_error_count = 0
    all_confidences, all_correct = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            little_out, big_out = model(inputs)

            confidence = torch.softmax(little_out, dim=1).max(dim=1).values
            little_pred = little_out.argmax(dim=1)
            big_pred = big_out.argmax(dim=1)
            use_little = confidence > threshold

            pred = torch.where(use_little, little_pred, big_pred)
            little_correct_mask = little_pred.eq(targets)
            big_correct_mask = big_pred.eq(targets)

            correct += pred.eq(targets).sum().item()
            little_correct += little_correct_mask.sum().item()
            big_correct += big_correct_mask.sum().item()
            exit_count += use_little.sum().item()
            routing_error_count += (
                use_little & ~little_correct_mask & big_correct_mask
            ).sum().item()
            total += inputs.size(0)

            all_confidences.append(confidence)
            all_correct.append(little_correct_mask)

    all_conf = torch.cat(all_confidences).cpu().numpy()
    all_corr = torch.cat(all_correct).cpu().numpy().astype(float)
    little_ece = compute_ece(all_conf, all_corr)

    return {
        "cascade_acc": correct / total,
        "little_acc": little_correct / total,
        "big_acc": big_correct / total,
        "exit_ratio": exit_count / total,
        "little_ece": little_ece,
        "routing_error_rate": routing_error_count / total,
    }


def proxy_train(
    genotype: PairGenotype,
    train_loader,
    val_loader,
    epochs: int = 10,
    lr: float = 0.05,
    device: torch.device | None = None,
    threshold: float = 0.8,
    label_smoothing: float = 0.0,
    little_loss_type: str = "ce",
    focal_gamma: float = 2.0,
    cascade_aware: bool = False,
    routing_sharpness: float = 20.0,
    little_aux_weight: float = 0.5,
    big_aux_weight: float = 0.0,
    defer_cost_weight: float = 0.01,
    confident_wrong_weight: float = 0.1,
    detach_routing_weight: bool = True,
) -> dict:
    """Co-train a CascadePair and return cascade metrics + memory."""
    if device is None:
        device = torch.device("cpu")

    model = CascadePair(genotype).to(device)
    n_params = count_parameters(model)
    memory = estimate_pair_memory(genotype)
    flops = estimate_pair_flops(genotype)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            little_out, big_out = model(inputs)
            if cascade_aware:
                loss = cascade_aware_joint_loss(
                    little_out, big_out, targets,
                    threshold=threshold,
                    label_smoothing=label_smoothing,
                    little_loss_type=little_loss_type,
                    focal_gamma=focal_gamma,
                    routing_sharpness=routing_sharpness,
                    little_aux_weight=little_aux_weight,
                    big_aux_weight=big_aux_weight,
                    defer_cost_weight=defer_cost_weight,
                    confident_wrong_weight=confident_wrong_weight,
                    detach_routing_weight=detach_routing_weight,
                )
            else:
                little_loss = branch_loss_per_sample(
                    little_out, targets, little_loss_type, label_smoothing, focal_gamma
                ).mean()
                big_loss = F.cross_entropy(big_out, targets, label_smoothing=label_smoothing)
                loss = little_loss + big_loss
            loss.backward()
            optimizer.step()

    metrics = evaluate_cascade(model, val_loader, device, threshold=threshold)
    metrics["n_params"] = n_params
    metrics["total_params"] = memory["total_params"]
    metrics["total_memory"] = memory["total_bytes"]
    metrics["little_flops"] = flops["little_flops"]
    metrics["big_flops"] = flops["big_flops"]
    metrics["cascade_flops"] = cascade_flops(flops["little_flops"], flops["big_flops"], metrics["exit_ratio"])
    return metrics
