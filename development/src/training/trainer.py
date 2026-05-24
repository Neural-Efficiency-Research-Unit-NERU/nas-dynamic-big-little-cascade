import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.training.utils import compute_ece


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal loss for calibration-aware training (Lin et al. 2017).

    Down-weights well-classified examples, focusing on hard samples.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    """
    return focal_loss_per_sample(logits, targets, gamma, label_smoothing).mean()


def focal_loss_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Per-sample focal loss, used when routing weights differ per example."""
    ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=label_smoothing)
    p_t = torch.exp(-ce)
    return (1.0 - p_t) ** gamma * ce


def branch_loss_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "ce",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Return one classification loss value per sample."""
    if loss_type == "focal":
        return focal_loss_per_sample(
            logits, targets, gamma=focal_gamma, label_smoothing=label_smoothing
        )
    if loss_type == "ce":
        return F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=label_smoothing
        )
    raise ValueError(f"Unknown loss_type: {loss_type!r}. Choose 'ce' or 'focal'.")


def cascade_aware_joint_loss(
    little_out: torch.Tensor,
    big_out: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    label_smoothing: float = 0.0,
    little_loss_type: str = "ce",
    focal_gamma: float = 2.0,
    routing_sharpness: float = 20.0,
    little_aux_weight: float = 0.5,
    big_aux_weight: float = 0.0,
    defer_cost_weight: float = 0.01,
    confident_wrong_weight: float = 0.1,
    detach_routing_weight: bool = True,
) -> torch.Tensor:
    """Differentiable proxy for cascade routing.

    `p_exit` is a soft version of confidence > threshold. Classification
    losses are weighted by the route a sample is likely to take, while two
    small confidence terms encourage cheap exits and discourage confident
    wrong exits.
    """
    little_loss = branch_loss_per_sample(
        little_out, targets, little_loss_type, label_smoothing, focal_gamma
    )
    big_loss = branch_loss_per_sample(
        big_out, targets, "ce", label_smoothing, focal_gamma
    )

    confidence = torch.softmax(little_out, dim=1).max(dim=1).values
    p_exit = torch.sigmoid(routing_sharpness * (confidence - threshold))
    route_weight = p_exit.detach() if detach_routing_weight else p_exit

    loss = (
        (route_weight + little_aux_weight) * little_loss
        + (1.0 - route_weight + big_aux_weight) * big_loss
    ).mean()

    if defer_cost_weight:
        loss = loss + defer_cost_weight * (1.0 - p_exit).mean()

    if confident_wrong_weight:
        wrong_little = little_out.argmax(dim=1).ne(targets).float()
        loss = loss + confident_wrong_weight * (p_exit * wrong_little).mean()

    return loss


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
    experiment_dir: Path,
    model_name: str = "model",
) -> dict:
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    momentum = config["training"].get("momentum", 0.9)
    weight_decay = config["training"].get("weight_decay", 5e-4)
    use_scheduler = config["training"].get("scheduler", None) == "cosine"

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) if use_scheduler else None

    csv_path = experiment_dir / f"{model_name}_history.csv"
    best_val_acc = 0.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_s"])

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                             f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}", f"{elapsed:.1f}"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), experiment_dir / f"{model_name}_best.pt")

        tqdm.write(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train {train_acc:.4f} | Val {val_acc:.4f} | "
            f"LR {current_lr:.4f} | {elapsed:.1f}s"
        )

    torch.save(model.state_dict(), experiment_dir / f"{model_name}_final.pt")
    return {"best_val_acc": best_val_acc, "csv_path": str(csv_path)}


def train_one_epoch_joint(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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
) -> tuple[float, float, float]:
    """Train paired model with joint loss. Returns (loss, little_acc, big_acc)."""
    model.train()
    total_loss, little_corr, big_corr, total = 0.0, 0, 0, 0
    for inputs, targets in loader:
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
        total_loss += loss.item() * inputs.size(0)
        little_corr += little_out.argmax(1).eq(targets).sum().item()
        big_corr += big_out.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, little_corr / total, big_corr / total


@torch.no_grad()
def evaluate_joint(
    model: nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.8,
) -> dict:
    """Evaluate shared backbone model. Returns dict with all metrics."""
    model.eval()
    total_loss, correct, little_corr, big_corr, exit_count, total = 0.0, 0, 0, 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    all_confidences, all_correct = [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        little_out, big_out = model(inputs)
        loss = criterion(little_out, targets) + criterion(big_out, targets)
        total_loss += loss.item() * inputs.size(0)

        confidence = torch.softmax(little_out, dim=1).max(dim=1).values
        little_pred = little_out.argmax(1)
        big_pred = big_out.argmax(1)
        use_little = confidence > threshold
        pred = torch.where(use_little, little_pred, big_pred)

        correct += pred.eq(targets).sum().item()
        little_corr += little_pred.eq(targets).sum().item()
        big_corr += big_pred.eq(targets).sum().item()
        exit_count += use_little.sum().item()
        total += inputs.size(0)

        all_confidences.append(confidence)
        all_correct.append(little_pred.eq(targets))

    all_conf = torch.cat(all_confidences).cpu().numpy()
    all_corr = torch.cat(all_correct).cpu().numpy().astype(float)
    little_ece = compute_ece(all_conf, all_corr)

    return {
        "loss": total_loss / total,
        "cascade_acc": correct / total,
        "little_acc": little_corr / total,
        "big_acc": big_corr / total,
        "exit_ratio": exit_count / total,
        "little_ece": little_ece,
    }


def train_joint_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
    experiment_dir: Path,
    model_name: str = "model",
    threshold: float = 0.8,
) -> dict:
    """Full training loop for shared backbone model with joint loss."""
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    momentum = config["training"].get("momentum", 0.9)
    weight_decay = config["training"].get("weight_decay", 5e-4)
    use_scheduler = config["training"].get("scheduler", None) == "cosine"

    co = config.get("co_training", {})
    label_smoothing = co.get("label_smoothing", 0.0)
    little_loss_type = co.get("little_loss_type", "ce")
    focal_gamma = co.get("focal_gamma", 2.0)
    cascade_aware = co.get("cascade_aware", False)
    routing_sharpness = co.get("routing_sharpness", 20.0)
    little_aux_weight = co.get("little_aux_weight", 0.5)
    big_aux_weight = co.get("big_aux_weight", 0.0)
    defer_cost_weight = co.get("defer_cost_weight", 0.01)
    confident_wrong_weight = co.get("confident_wrong_weight", 0.1)
    detach_routing_weight = co.get("detach_routing_weight", True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) if use_scheduler else None

    csv_path = experiment_dir / f"{model_name}_history.csv"
    best_cascade_acc = 0.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "little_acc", "big_acc",
            "val_loss", "val_cascade_acc", "val_little_acc", "val_big_acc",
            "val_exit_ratio", "val_little_ece", "lr", "time_s",
        ])

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_little, train_big = train_one_epoch_joint(
            model, train_loader, optimizer, device,
            threshold=threshold,
            label_smoothing=label_smoothing, little_loss_type=little_loss_type,
            focal_gamma=focal_gamma,
            cascade_aware=cascade_aware,
            routing_sharpness=routing_sharpness,
            little_aux_weight=little_aux_weight,
            big_aux_weight=big_aux_weight,
            defer_cost_weight=defer_cost_weight,
            confident_wrong_weight=confident_wrong_weight,
            detach_routing_weight=detach_routing_weight,
        )
        val_metrics = evaluate_joint(model, val_loader, device, threshold=threshold)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.4f}", f"{train_little:.4f}", f"{train_big:.4f}",
                f"{val_metrics['loss']:.4f}", f"{val_metrics['cascade_acc']:.4f}",
                f"{val_metrics['little_acc']:.4f}", f"{val_metrics['big_acc']:.4f}",
                f"{val_metrics['exit_ratio']:.2f}", f"{val_metrics['little_ece']:.4f}",
                f"{current_lr:.6f}", f"{elapsed:.1f}",
            ])

        if val_metrics["cascade_acc"] > best_cascade_acc:
            best_cascade_acc = val_metrics["cascade_acc"]
            torch.save(model.state_dict(), experiment_dir / f"{model_name}_best.pt")

        tqdm.write(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Cascade {val_metrics['cascade_acc']:.4f} | "
            f"Little {val_metrics['little_acc']:.4f} | Big {val_metrics['big_acc']:.4f} | "
            f"Exit {val_metrics['exit_ratio']:.2f} | ECE {val_metrics['little_ece']:.4f} | "
            f"LR {current_lr:.4f} | {elapsed:.1f}s"
        )

    torch.save(model.state_dict(), experiment_dir / f"{model_name}_final.pt")
    return {"best_cascade_acc": best_cascade_acc, "csv_path": str(csv_path)}
