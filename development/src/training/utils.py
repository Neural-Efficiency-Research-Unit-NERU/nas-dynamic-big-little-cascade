import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE).

    Bins predictions by confidence, computes |accuracy - confidence| per bin,
    returns weighted average. Perfect calibration => ECE = 0.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidences)
    if total == 0:
        return 0.0

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            # Last bin is inclusive on both ends
            mask = (confidences >= lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (count / total) * abs(avg_acc - avg_conf)
    return float(ece)


def learn_temperature(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
) -> float:
    """Learn optimal temperature for calibration on the validation set.

    Guo et al., "On Calibration of Modern Neural Networks" (2017).
    Minimizes NLL with a single learned temperature scalar.
    Returns the learned temperature (float > 0).
    """
    model.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            little_out, _ = model(inputs)
            all_logits.append(little_out)
            all_labels.append(targets)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    temperature = torch.nn.Parameter(torch.ones(1, device=device) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def get_experiment_dir(experiment_name: str) -> Path:
    exp_dir = Path(__file__).resolve().parent.parent.parent / "experiments" / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


BYTES_PER_PARAM = 4  # float32

# Hard deployment-memory constraint (450 KiB).
# Fits 512 KB MCU SRAM with ~60 KB headroom for runtime/stack.
MEMORY_BUDGET_BYTES = 460_800

# Default hard parameter cap (~equivalent to 400 KB of float32 weights).
# The NAS also enforces the deployment-memory cap above; both must pass.
PARAM_BUDGET_REFERENCE = 100_000


def budget_utilisation(total_bytes: int, budget: int = MEMORY_BUDGET_BYTES) -> float:
    """Fraction of the memory budget consumed.

    Closer to 1.0 means the candidate uses more of the available deployment
    memory budget.
    """
    return total_bytes / budget


def _estimate_path_memory(
    blocks: list, input_channels: int = 3
) -> tuple[int, int]:
    """Estimate weight bytes and peak activation bytes for one model path.

    All blocks use stride=2. Returns (weight_bytes, peak_activation_bytes) in float32.
    """
    total_params = 0
    peak_activation = 0
    h, w = 32, 32
    in_ch = input_channels

    for i, block in enumerate(blocks):
        stride = 2  # all blocks downsample
        out_ch = block.channels
        first_c_in = in_ch  # for residual shortcut check

        for j in range(block.layers):
            s = stride if j == 0 else 1
            c_in = in_ch if j == 0 else out_ch
            if s == 2:
                h, w = h // 2, w // 2

            if block.conv_type == "depthwise_separable":
                total_params += c_in * block.kernel_size * block.kernel_size
                total_params += c_in * 2  # BN
                total_params += c_in * out_ch
                total_params += out_ch * 2  # BN
            else:
                total_params += c_in * out_ch * block.kernel_size * block.kernel_size
                total_params += out_ch * 2  # BN

            peak_activation = max(peak_activation, h * w * out_ch)
            in_ch = out_ch

        # Residual shortcut params (1x1 conv + BN when channel/stride mismatch)
        if block.use_residual and (first_c_in != out_ch or stride != 1):
            total_params += first_c_in * out_ch  # 1x1 conv
            total_params += out_ch * 2  # BN

    # Classifier head: Linear(last_ch -> 10)
    last_ch = blocks[-1].channels
    total_params += last_ch * 10 + 10  # weights + bias

    return total_params * BYTES_PER_PARAM, peak_activation * BYTES_PER_PARAM


def _estimate_path_flops(
    blocks: list, input_channels: int = 3, num_classes: int = 10
) -> int:
    """Estimate multiply-accumulate operations (MACs) for one model path.

    All blocks use stride=2. BN excluded (fused at inference).
    """
    total_macs = 0
    h, w = 32, 32
    in_ch = input_channels

    for i, block in enumerate(blocks):
        stride = 2  # all blocks downsample
        out_ch = block.channels
        first_c_in = in_ch  # for residual shortcut check

        for j in range(block.layers):
            s = stride if j == 0 else 1
            c_in = in_ch if j == 0 else out_ch
            if s == 2:
                h, w = h // 2, w // 2

            if block.conv_type == "depthwise_separable":
                # Depthwise: in_ch * k * k * H_out * W_out
                total_macs += c_in * block.kernel_size * block.kernel_size * h * w
                # Pointwise: in_ch * out_ch * H_out * W_out
                total_macs += c_in * out_ch * h * w
            else:
                # Standard conv: out_ch * in_ch * k * k * H_out * W_out
                total_macs += out_ch * c_in * block.kernel_size * block.kernel_size * h * w

            in_ch = out_ch

        # Residual 1x1 shortcut MACs (when channel/stride mismatch)
        if block.use_residual and (first_c_in != out_ch or stride != 1):
            # 1x1 conv: in_ch * out_ch * H_out * W_out
            # h,w already reflect the stride from the first layer
            total_macs += first_c_in * out_ch * h * w

    # Classifier head: Linear(last_ch -> num_classes)
    last_ch = blocks[-1].channels
    total_macs += last_ch * num_classes

    return total_macs


def estimate_pair_flops(genotype, num_classes: int = 10) -> dict:
    """Analytically estimate FLOPs (MACs) for a CascadePair."""
    from src.nas.search_space import INPUT_CHANNELS

    little_flops = _estimate_path_flops(
        genotype.little_blocks, INPUT_CHANNELS, num_classes
    )
    big_flops = _estimate_path_flops(
        genotype.big_blocks, INPUT_CHANNELS, num_classes
    )

    return {
        "little_flops": little_flops,
        "big_flops": big_flops,
        "total_flops": little_flops + big_flops,
    }


def cascade_flops(little_flops: int, big_flops: int, exit_ratio: float) -> float:
    """Expected FLOPs for cascade inference.

    All samples run little model. (1 - exit_ratio) fraction also runs big model.
    """
    return little_flops + (1.0 - exit_ratio) * big_flops


def estimate_pair_memory(genotype) -> dict:
    """Analytically estimate deployment memory and parameter count for a PairGenotype.

    Returns both byte-level memory (float32) and raw parameter counts.
    The param count is the primary constraint for NAS (< 100K).
    """
    from src.nas.search_space import INPUT_CHANNELS

    little_weight, little_peak = _estimate_path_memory(
        genotype.little_blocks, INPUT_CHANNELS
    )
    big_weight, big_peak = _estimate_path_memory(
        genotype.big_blocks, INPUT_CHANNELS
    )

    input_bytes = 32 * 32 * INPUT_CHANNELS * BYTES_PER_PARAM

    total = (
        little_weight
        + big_weight
        + max(little_peak, big_peak)
        + input_bytes
    )

    little_params = little_weight // BYTES_PER_PARAM
    big_params = big_weight // BYTES_PER_PARAM
    total_params = little_params + big_params

    return {
        "little_weight_bytes": little_weight,
        "big_weight_bytes": big_weight,
        "little_peak_activation": little_peak,
        "big_peak_activation": big_peak,
        "input_bytes": input_bytes,
        "total_bytes": total,
        "little_params": little_params,
        "big_params": big_params,
        "total_params": total_params,
    }


@torch.no_grad()
def compute_nll(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    temperature: float = 1.0,
) -> float:
    """Compute negative log-likelihood of the little model on a dataset."""
    model.eval()
    total_nll = 0.0
    total = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        little_out, _ = model(inputs)
        total_nll += criterion(little_out / temperature, targets).item()
        total += inputs.size(0)
    return total_nll / total




@torch.no_grad()
def evaluate_at_thresholds(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    thresholds: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    temperature: float = 1.0,
) -> list[dict]:
    """Evaluate cascade at multiple thresholds in a single pass.

    Returns list of dicts per threshold, each containing:
      threshold, cascade_acc, exit_ratio, little_acc, big_acc,
      little_ece, routing_error_rate.

    routing_error_rate is the fraction of samples the little model exits with a
    wrong prediction while the big model would have produced the correct label.
    This is a cascade-routing regret metric; lower is better.
    """
    model.eval()
    all_little_out = []
    all_big_out = []
    all_targets = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        little_out, big_out = model(inputs)
        all_little_out.append(little_out.cpu())
        all_big_out.append(big_out.cpu())
        all_targets.append(targets.cpu())

    little_logits = torch.cat(all_little_out)
    big_logits = torch.cat(all_big_out)
    targets = torch.cat(all_targets)

    confidence = torch.softmax(little_logits / temperature, dim=1).max(dim=1).values
    little_pred = little_logits.argmax(dim=1)
    big_pred = big_logits.argmax(dim=1)
    little_correct = little_pred.eq(targets)
    big_correct = big_pred.eq(targets)
    n = targets.size(0)

    results = []
    for t in thresholds:
        use_little = confidence > t
        pred = torch.where(use_little, little_pred, big_pred)
        cascade_acc = pred.eq(targets).float().mean().item()
        exit_ratio = use_little.float().mean().item()
        little_acc = little_correct.float().mean().item()
        big_acc = big_correct.float().mean().item()

        # Routing regret: little exits, little wrong, big would be right.
        routing_error = (use_little & ~little_correct & big_correct).float().sum().item() / n

        conf_np = confidence.numpy()
        corr_np = little_correct.numpy().astype(float)
        ece = compute_ece(conf_np, corr_np)

        results.append({
            "threshold": t,
            "cascade_acc": cascade_acc,
            "exit_ratio": exit_ratio,
            "little_acc": little_acc,
            "big_acc": big_acc,
            "little_ece": ece,
            "routing_error_rate": routing_error,
        })
    return results


@torch.no_grad()
def collect_confidence_data(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect little model confidence and correctness for calibration analysis.

    Returns (confidences, correctness) arrays.
    """
    model.eval()
    all_conf = []
    all_corr = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        little_out, _ = model(inputs)
        probs = torch.softmax(little_out / temperature, dim=1)
        confidence = probs.max(dim=1).values
        correct = little_out.argmax(dim=1).eq(targets).float()
        all_conf.append(confidence.cpu().numpy())
        all_corr.append(correct.cpu().numpy())
    return np.concatenate(all_conf), np.concatenate(all_corr)
