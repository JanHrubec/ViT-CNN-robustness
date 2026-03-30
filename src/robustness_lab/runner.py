from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config_schema import DatasetConfig
from .corruptions import CorruptionSpec, make_clean_transform, make_corruption_transform
from .datasets import TransformedDataset, build_loader
from .metrics import topk_accuracy_from_state, update_topk_correct


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


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    topk: tuple[int, ...],
    desc: str,
) -> dict[str, float]:
    """Shared inference loop used by both clean and corrupted evaluation."""
    state: dict[int, int] = {k: 0 for k in topk}
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc, leave=False):
            # Non-blocking transfer helps when using pinned CPU memory.
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)

            update_topk_correct(logits, targets, topk, state)
            total += targets.shape[0]

    metrics = topk_accuracy_from_state(state, total)
    metrics["count"] = total
    return metrics


def evaluate_clean(
    model: torch.nn.Module,
    model_name: str,
    preprocess: Callable,
    base_dataset: Dataset,
    dataset_cfg: DatasetConfig,
    device: torch.device,
    topk: tuple[int, ...],
) -> EvalResult:
    """Evaluate model on uncorrupted images (baseline)."""
    dataset = TransformedDataset(base_dataset, make_clean_transform(preprocess))
    loader = build_loader(dataset, dataset_cfg)
    m = _evaluate_loader(model, loader, device, topk, desc=f"{model_name}: clean")

    return EvalResult(
        model=model_name,
        split="clean",
        corruption_family="none",
        corruption_name="none",
        severity=0.0,
        sample_count=int(m["count"]),
        top1=float(m.get("top1", 0.0)),
        top5=float(m.get("top5", 0.0)),
    )


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
) -> EvalResult:
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

    return EvalResult(
        model=model_name,
        split="corrupted",
        corruption_family=spec.family,
        corruption_name=spec.name,
        severity=float(spec.severity),
        sample_count=int(m["count"]),
        top1=float(m.get("top1", 0.0)),
        top5=float(m.get("top5", 0.0)),
    )
