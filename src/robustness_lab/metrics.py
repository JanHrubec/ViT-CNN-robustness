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
    """Update running top-k counters"""
    max_k = max(ks)
    _, pred = logits.topk(max_k, dim=1)
    pred_t = pred.t()
    correct = pred_t.eq(targets.view(1, -1).expand_as(pred_t))

    for k in ks:
        hits = correct[:k].any(dim=0).sum().item()
        state[k] = state.get(k, 0) + int(hits)


def topk_hits_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: Iterable[int],
) -> dict[int, torch.Tensor]:
    """Per-sample binary hit tensors for each top-k"""
    max_k = max(ks)
    _, pred = logits.topk(max_k, dim=1)
    correct = pred.eq(targets.view(-1, 1))
    out: dict[int, torch.Tensor] = {}
    for k in ks:
        out[k] = correct[:, :k].any(dim=1)
    return out


def negative_log_likelihood_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Per-sample negative log-likelihood values."""
    log_probs = torch.log_softmax(logits, dim=1)
    return -log_probs.gather(1, targets.view(-1, 1)).squeeze(1)


class ECEAccumulator:
    """Expected Calibration Error accumulator."""

    def __init__(self, bins: int = 15, device: torch.device | None = None) -> None:
        self.bins = bins
        self.device = device
        self.edges = torch.linspace(0, 1, bins + 1, device=device)
        self.counts = torch.zeros(bins, device=device)
        self.conf_sums = torch.zeros(bins, device=device)
        self.acc_sums = torch.zeros(bins, device=device)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        acc = pred.eq(targets).float()

        # Bin by confidence
        for i in range(self.bins):
            left, right = self.edges[i], self.edges[i + 1]
            mask = (conf > left) & (conf <= right)
            if mask.any():
                bin_count = mask.float().sum()
                self.counts[i] += bin_count
                self.conf_sums[i] += conf[mask].sum()
                self.acc_sums[i] += acc[mask].sum()

    def compute(self) -> float:
        total = self.counts.sum().clamp_min(1.0)
        avg_conf = torch.where(self.counts > 0, self.conf_sums / self.counts, 0.0)
        avg_acc = torch.where(self.counts > 0, self.acc_sums / self.counts, 0.0)
        ece = (self.counts / total) * torch.abs(avg_acc - avg_conf)
        return float(ece.sum().item())


def topk_accuracy_from_state(state: dict[int, int], total: int) -> dict[str, float]:
    """Accumulated counts to top-k accuracies"""
    if total <= 0:
        return {f"top{k}": 0.0 for k in state}
    return {f"top{k}": state[k] / total for k in sorted(state)}


def robustness_ratio(clean_acc: float, corrupt_acc: float) -> float:
    """Corrupted accuracy normalized by clean accuracy."""
    if clean_acc <= 0:
        return 0.0
    return corrupt_acc / clean_acc


def audc(severities: list[float], accuracies: list[float]) -> float:
    """Normalized area under accuracy-vs-severity curve. Higher value better"""
    if len(severities) < 2:
        return 0.0
    x = np.asarray(severities, dtype=float)
    y = np.asarray(accuracies, dtype=float)
    area = np.trapezoid(y, x)
    denom = max(x.max() - x.min(), 1e-12)
    return float(area / denom)


def linear_trend_slope(severities: list[float], values: list[float]) -> float:
    """Slope of line fitted to value vs severity."""
    if len(severities) < 2:
        return 0.0
    x = np.asarray(severities, dtype=float)
    y = np.asarray(values, dtype=float)
    m, _ = np.polyfit(x, y, deg=1)
    return float(m)


def endpoint_delta(reference: float, at_max_severity: float) -> float:
    """Difference between clean and corrupted at highest severity"""
    return float(at_max_severity - reference)


def expected_calibration_error(logits: torch.Tensor, targets: torch.Tensor, bins: int = 15) -> float:
    """Classic ECE with equal-width confidence bins"""
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    acc = pred.eq(targets)

    edges = torch.linspace(0, 1, bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)

    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (conf > left) & (conf <= right)
        if mask.any():
            bin_acc = acc[mask].float().mean()
            bin_conf = conf[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_acc - bin_conf)

    return float(ece.item())

def compute_prediction_stability(per_sample_rows: Iterable[dict]) -> list[dict]:
    """
    Prediction stability relative to clean

    Fraction of samples whose top-1 prediction matches clean prediction for the same model and dataset index
    """
    clean_pred: dict[tuple[str, int], int] = {}

    for row in per_sample_rows:
        if row.get("split") != "clean":
            continue
        model = row.get("model")
        idx = row.get("sample_index")
        pred = row.get("top1_pred")
        if model is None or idx is None or pred is None:
            continue
        clean_pred[(model, int(idx))] = int(pred)

    grouped: dict[tuple[str, str, float], list[int]] = defaultdict(list)

    for row in per_sample_rows:
        if row.get("split") != "corrupted":
            continue
        model = row.get("model")
        idx = row.get("sample_index")
        pred = row.get("top1_pred")
        family = row.get("corruption_family")
        severity = row.get("severity")
        if model is None or idx is None or pred is None or family is None or severity is None:
            continue
        key = (str(model), str(family), float(severity))
        clean = clean_pred.get((model, int(idx)))
        if clean is None:
            continue
        grouped[key].append(1 if int(pred) == clean else 0)

    out: list[dict] = []
    for (model, family, severity), hits in sorted(grouped.items()):
        if not hits:
            continue
        out.append(
            {
                "model": model,
                "corruption_family": family,
                "severity": severity,
                "stability_top1": sum(hits) / len(hits),
                "sample_count": len(hits),
            }
        )

    return out


def bootstrap_ci(
    values: list[float],
    iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Confidence interval for the mean of scalar values."""
    if not values:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means = []

    for _ in range(iters):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(sample.mean())

    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)
