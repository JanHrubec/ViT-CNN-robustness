from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import multiprocessing as mp
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR100

from .config_schema import DatasetConfig


class TransformedDataset(Dataset):
    """Thin wrapper: keeps labels from base dataset, swaps transform dynamically.

    When `return_index` is True, __getitem__ returns (image, target, index),
    which enables paired analyses across corruptions.
    """

    def __init__(self, base: Dataset, transform: Callable, return_index: bool = False):
        self.base = base
        self.transform = transform
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, target = self.base[idx]
        image = self.transform(image)
        if self.return_index:
            return image, target, idx
        return image, target


def _subset_per_class(dataset: Dataset, per_class: int, seed: int) -> Subset:
    """Create deterministic balanced subset by sampling fixed count per class."""
    g = torch.Generator().manual_seed(seed)
    by_class: dict[int, list[int]] = defaultdict(list)

    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Dataset does not expose `targets`, cannot build per-class subset.")

    for i, y in enumerate(targets):
        by_class[int(y)].append(i)

    selected: list[int] = []
    for cls in sorted(by_class):
        # Shuffle within class with seeded generator for repeatable subsets.
        cls_indices = torch.tensor(by_class[cls], dtype=torch.long)
        perm = cls_indices[torch.randperm(len(cls_indices), generator=g)]
        take = min(per_class, len(perm))
        selected.extend(perm[:take].tolist())

    return Subset(dataset, selected)


def build_base_dataset(cfg: DatasetConfig, seed: int = 42) -> Dataset:
    """Build untransformed base dataset; transforms are attached later per condition."""
    root = Path(cfg.root).expanduser()

    if cfg.name == "cifar100":
        dataset = CIFAR100(root=str(root), train=False, download=True, transform=None)
    else:
        raise ValueError("Only cifar100 is supported in this project.")

    if cfg.subset_per_class is not None:
        dataset = _subset_per_class(dataset, cfg.subset_per_class, seed=seed)

    return dataset


def build_loader(dataset: Dataset, cfg: DatasetConfig) -> DataLoader:
    """Build evaluation dataloader with stable ordering (`shuffle=False`)."""
    num_workers = cfg.num_workers
    start_method = mp.get_start_method(allow_none=True)
    if num_workers > 0 and start_method in (None, "spawn", "forkserver"):
        # Local transform closures are not picklable under spawn; use single worker.
        num_workers = 0
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )
