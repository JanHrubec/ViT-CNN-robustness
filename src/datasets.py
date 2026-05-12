from __future__ import annotations

import io
import pickle
import random
from pathlib import Path
import multiprocessing as mp
from typing import Any, Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config_schema import DatasetConfig

IMAGENET_VAL_MAX_PER_CLASS = 50


class TransformedDataset(Dataset):
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


class InMemoryImageDataset(Dataset):
    def __init__(self, items: list[tuple[bytes, int]]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        data, label = self._items[idx]
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img, int(label)


def jpeg_encode(image: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def stream_imagenet_pools(pool_per_class: int, num_classes: int = 1000) -> dict[int, list[bytes]]:
    from datasets import load_dataset

    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        streaming=True,
    )
    buckets: dict[int, list[bytes]] = {}
    for example in ds:
        label = int(example["label"])
        if label not in buckets:
            buckets[label] = []
        if len(buckets[label]) < pool_per_class:
            pil = example["image"]
            if not isinstance(pil, Image.Image):
                pil = Image.fromarray(pil)
            buckets[label].append(jpeg_encode(pil))
        if len(buckets) == num_classes and all(
            len(buckets.get(c, ())) >= pool_per_class for c in range(num_classes)
        ):
            break

    missing = [c for c in range(num_classes) if len(buckets.get(c, ())) < pool_per_class]
    if missing:
        raise RuntimeError(
            f"Streaming ended before filling all classes; first missing: {missing[:5]!r}."
        )
    return {k: buckets[k] for k in range(num_classes)}


def pool_cache_path(cache_dir: Path, pool_per_class: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"imagenet_val_pools_{pool_per_class}.pkl"


def ensure_imagenet_pools(cache_dir: Path | str, pool_per_class: int) -> dict[int, list[bytes]]:
    cdir = Path(cache_dir).expanduser()
    path = pool_cache_path(cdir, pool_per_class)
    if path.is_file():
        with path.open("rb") as f:
            payload: dict[str, Any] = pickle.load(f)
        if payload.get("format_version") != 1 or payload.get("pool_per_class") != pool_per_class:
            raise ValueError(f"Cache {path} is incompatible; delete it or fix pool size.")
        raw: dict[int | str, list[bytes]] = payload["buckets"]
        buckets: dict[int, list[bytes]] = {int(k): v for k, v in raw.items()}
    else:
        buckets = stream_imagenet_pools(pool_per_class)
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(
                {
                    "format_version": 1,
                    "pool_per_class": pool_per_class,
                    "buckets": buckets,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp.replace(path)

    for c in range(1000):
        if c not in buckets or len(buckets[c]) < pool_per_class:
            raise RuntimeError(f"Incomplete pool for class {c}")
    return buckets


def sample_imagenet_subset(buckets: dict[int, list[bytes]], n_per_class: int, seed: int) -> list[tuple[bytes, int]]:
    rng = random.Random(seed)
    items: list[tuple[bytes, int]] = []
    for label in sorted(buckets.keys()):
        pool = buckets[label]
        if len(pool) < n_per_class:
            raise ValueError(
                f"Pool {label} has {len(pool)}; need {n_per_class}. "
            )
        idxs = list(range(len(pool)))
        rng.shuffle(idxs)
        chosen_idx = sorted(idxs[:n_per_class])
        for i in chosen_idx:
            items.append((pool[i], label))
    return items


def build_base_dataset(cfg: DatasetConfig, seed: int = 42) -> Dataset:
    if not cfg.cache_dir:
        raise ValueError("dataset.cache_dir is required (directory for ImageNet subset cache).")
    if cfg.n_per_class <= 0:
        raise ValueError("dataset.n_per_class must be > 0")

    desired_pool = int(cfg.pool_per_class)
    if desired_pool <= int(cfg.n_per_class) and int(cfg.n_per_class) < IMAGENET_VAL_MAX_PER_CLASS:
        desired_pool = min(IMAGENET_VAL_MAX_PER_CLASS, int(cfg.n_per_class) + 1)

    pool_pc = min(desired_pool, IMAGENET_VAL_MAX_PER_CLASS)
    if pool_pc < cfg.n_per_class:
        raise ValueError(
            f"pool_per_class ({cfg.pool_per_class}) must be >= n_per_class ({cfg.n_per_class})."
        )

    cache_path = Path(cfg.cache_dir).expanduser()
    buckets = ensure_imagenet_pools(cache_path, pool_per_class=pool_pc)
    items = sample_imagenet_subset(buckets, cfg.n_per_class, seed=seed)
    return InMemoryImageDataset(items)


def build_loader(dataset: Dataset, cfg: DatasetConfig) -> DataLoader:
    num_workers = cfg.num_workers
    start_method = mp.get_start_method(allow_none=True)
    if num_workers > 0 and start_method in (None, "spawn", "forkserver"):
        num_workers = 0
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )
