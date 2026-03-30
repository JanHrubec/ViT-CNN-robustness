from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seeds across common RNG sources for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve target device with sensible fallback order."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return it as Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
