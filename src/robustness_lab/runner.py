from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config_schema import DatasetConfig
from .corruptions import CorruptionSpec, make_clean_transform, make_corruption_transform
from .datasets import TransformedDataset, build_loader
from .metrics import bootstrap_ci, topk_accuracy_from_state, topk_hits_per_sample


@dataclass
class EvalResult:
    """Standardized evaluation record for one model-condition pair."""
    model: str
    split: str
    corruption_family: str
    corruption_name: str
    severity: float
    sample_count: int
    top1: float
    top5: float
    top1_ci_low: float
    top1_ci_high: float
    top5_ci_low: float
    top5_ci_high: float


@dataclass
class EvalOutcome:
    """Evaluation result plus optional per-sample correctness rows."""
    result: EvalResult
    per_sample_rows: list[dict]


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    topk: tuple[int, ...],
    desc: str,
) -> dict:
    """Shared inference loop used by both clean and corrupted evaluation."""
    state: dict[int, int] = {k: 0 for k in topk}
    per_sample_hits: dict[int, list[int]] = {k: [] for k in topk}
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc, leave=False):
            # Non-blocking transfer helps when using pinned CPU memory.
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)

            hit_map = topk_hits_per_sample(logits, targets, topk)
            for k in topk:
                batch_hits = hit_map[k].to(torch.int32).cpu().tolist()
                state[k] += int(sum(batch_hits))
                per_sample_hits[k].extend(batch_hits)

            total += targets.shape[0]

    metrics = topk_accuracy_from_state(state, total)
    metrics["count"] = total
    metrics["hits"] = per_sample_hits
    return metrics


def evaluate_clean(
    model: torch.nn.Module,
    model_name: str,
    preprocess: Callable,
    base_dataset: Dataset,
    dataset_cfg: DatasetConfig,
    device: torch.device,
    topk: tuple[int, ...],
    bootstrap_iters: int,
    seed: int,
    save_per_sample: bool,
) -> EvalOutcome:
    """Evaluate model on uncorrupted images (baseline)."""
    dataset = TransformedDataset(base_dataset, make_clean_transform(preprocess))
    loader = build_loader(dataset, dataset_cfg)
    m = _evaluate_loader(model, loader, device, topk, desc=f"{model_name}: clean")

    top1_hits = [float(v) for v in m["hits"].get(1, [])]
    top5_hits = [float(v) for v in m["hits"].get(5, [])]
    top1_ci = bootstrap_ci(top1_hits, iters=bootstrap_iters, seed=seed)
    top5_ci = bootstrap_ci(top5_hits, iters=bootstrap_iters, seed=seed + 1)

    result = EvalResult(
        model=model_name,
        split="clean",
        corruption_family="none",
        corruption_name="none",
        severity=0.0,
        sample_count=int(m["count"]),
        top1=float(m.get("top1", 0.0)),
        top5=float(m.get("top5", 0.0)),
        top1_ci_low=top1_ci[0],
        top1_ci_high=top1_ci[1],
        top5_ci_low=top5_ci[0],
        top5_ci_high=top5_ci[1],
    )

    per_sample_rows: list[dict] = []
    if save_per_sample:
        n = int(m["count"])
        h1 = m["hits"].get(1, [0] * n)
        h5 = m["hits"].get(5, [0] * n)
        for i in range(n):
            per_sample_rows.append(
                {
                    "model": model_name,
                    "split": "clean",
                    "corruption_family": "none",
                    "corruption_name": "none",
                    "severity": 0.0,
                    "sample_index": i,
                    "top1_correct": int(h1[i]),
                    "top5_correct": int(h5[i]),
                }
            )

    return EvalOutcome(result=result, per_sample_rows=per_sample_rows)


def evaluate_corruption(
    model: torch.nn.Module,
    model_name: str,
    preprocess: Callable,
    base_dataset: Dataset,
    dataset_cfg: DatasetConfig,
    device: torch.device,
    topk: tuple[int, ...],
    spec: CorruptionSpec,
    seed: int,
    bootstrap_iters: int,
    save_per_sample: bool,
) -> EvalOutcome:
    """Evaluate model under one specific corruption setting."""
    transform = make_corruption_transform(spec, preprocess, seed=seed)
    dataset = TransformedDataset(base_dataset, transform)
    loader = build_loader(dataset, dataset_cfg)

    m = _evaluate_loader(
        model,
        loader,
        device,
        topk,
        desc=f"{model_name}: {spec.name}",
    )

    top1_hits = [float(v) for v in m["hits"].get(1, [])]
    top5_hits = [float(v) for v in m["hits"].get(5, [])]
    top1_ci = bootstrap_ci(top1_hits, iters=bootstrap_iters, seed=seed)
    top5_ci = bootstrap_ci(top5_hits, iters=bootstrap_iters, seed=seed + 1)

    result = EvalResult(
        model=model_name,
        split="corrupted",
        corruption_family=spec.family,
        corruption_name=spec.name,
        severity=float(spec.severity),
        sample_count=int(m["count"]),
        top1=float(m.get("top1", 0.0)),
        top5=float(m.get("top5", 0.0)),
        top1_ci_low=top1_ci[0],
        top1_ci_high=top1_ci[1],
        top5_ci_low=top5_ci[0],
        top5_ci_high=top5_ci[1],
    )

    per_sample_rows: list[dict] = []
    if save_per_sample:
        n = int(m["count"])
        h1 = m["hits"].get(1, [0] * n)
        h5 = m["hits"].get(5, [0] * n)
        for i in range(n):
            per_sample_rows.append(
                {
                    "model": model_name,
                    "split": "corrupted",
                    "corruption_family": spec.family,
                    "corruption_name": spec.name,
                    "severity": float(spec.severity),
                    "sample_index": i,
                    "top1_correct": int(h1[i]),
                    "top5_correct": int(h5[i]),
                }
            )

    return EvalOutcome(result=result, per_sample_rows=per_sample_rows)
