from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def update_topk_correct(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: Iterable[int],
    state: dict[int, int],
) -> None:
    """Update running top-k correct counters for a batch."""
    max_k = max(ks)
    _, pred = logits.topk(max_k, dim=1)
    pred_t = pred.t()
    correct = pred_t.eq(targets.view(1, -1).expand_as(pred_t))

    for k in ks:
        # Count sample as correct if ground truth appears anywhere in top-k.
        hits = correct[:k].any(dim=0).sum().item()
        state[k] = state.get(k, 0) + int(hits)


def topk_hits_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: Iterable[int],
) -> dict[int, torch.Tensor]:
    """Return per-sample binary hit tensors for each requested top-k."""
    max_k = max(ks)
    _, pred = logits.topk(max_k, dim=1)
    correct = pred.eq(targets.view(-1, 1))
    out: dict[int, torch.Tensor] = {}
    for k in ks:
        out[k] = correct[:, :k].any(dim=1)
    return out


def topk_accuracy_from_state(state: dict[int, int], total: int) -> dict[str, float]:
    """Convert accumulated counts into top-k accuracies."""
    if total <= 0:
        return {f"top{k}": 0.0 for k in state}
    return {f"top{k}": state[k] / total for k in sorted(state)}


def robustness_ratio(clean_acc: float, corrupt_acc: float) -> float:
    """Relative robustness: corrupted accuracy normalized by clean accuracy."""
    if clean_acc <= 0:
        return 0.0
    return corrupt_acc / clean_acc


def audc(severities: list[float], accuracies: list[float]) -> float:
    """Normalized area under accuracy-vs-severity curve.

    Higher value means the model keeps more accuracy as corruption increases.
    """
    if len(severities) < 2:
        return 0.0
    x = np.asarray(severities, dtype=float)
    y = np.asarray(accuracies, dtype=float)
    area = np.trapz(y, x)
    denom = max(x.max() - x.min(), 1e-12)
    return float(area / denom)


def expected_calibration_error(logits: torch.Tensor, targets: torch.Tensor, bins: int = 15) -> float:
    """Compute classic ECE using equal-width confidence bins."""
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    acc = pred.eq(targets)

    edges = torch.linspace(0, 1, bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)

    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (conf > left) & (conf <= right)
        if mask.any():
            # Weighted absolute gap between confidence and empirical accuracy.
            bin_acc = acc[mask].float().mean()
            bin_conf = conf[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_acc - bin_conf)

    return float(ece.item())


def bootstrap_ci(
    values: list[float],
    iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean of scalar values."""
    if not values:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means = []

    for _ in range(iters):
        # Resample with replacement, then recompute mean.
        sample = rng.choice(arr, size=n, replace=True)
        means.append(sample.mean())

    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)
