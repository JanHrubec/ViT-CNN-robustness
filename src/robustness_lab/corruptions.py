from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from .config_schema import CorruptionsConfig


@dataclass(frozen=True)
class CorruptionSpec:
    family: str
    severity: float
    name: str


def build_corruption_specs(cfg: CorruptionsConfig) -> list[CorruptionSpec]:
    specs: list[CorruptionSpec] = []

    for deg in cfg.rotation_degrees:
        specs.append(CorruptionSpec(family="rotation", severity=float(deg), name=f"rot_{deg}"))

    for px in cfg.translation_pixels:
        specs.append(CorruptionSpec(family="translation_x", severity=float(px), name=f"tx_{px}"))
        specs.append(CorruptionSpec(family="translation_y", severity=float(px), name=f"ty_{px}"))

    for sigma in cfg.gaussian_sigmas:
        specs.append(CorruptionSpec(family="gaussian_noise", severity=float(sigma), name=f"gauss_{sigma}"))

    return specs


def _apply_rotation(x: torch.Tensor, deg: float) -> torch.Tensor:
    return F.rotate(x, angle=deg, interpolation=InterpolationMode.BILINEAR, fill=0)


def _apply_translation_x(x: torch.Tensor, px: int) -> torch.Tensor:
    return F.affine(
        x,
        angle=0.0,
        translate=[px, 0],
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )


def _apply_translation_y(x: torch.Tensor, px: int) -> torch.Tensor:
    return F.affine(
        x,
        angle=0.0,
        translate=[0, px],
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )


def _apply_gaussian_noise(x: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    if sigma <= 0:
        return x
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)


def make_corruption_transform(
    spec: CorruptionSpec,
    preprocess: Callable,
    seed: int,
) -> Callable:
    """
    Returns callable that expects a PIL image and outputs preprocessed tensor.
    """

    generator = torch.Generator().manual_seed(seed)

    def _transform(img) -> torch.Tensor:
        x = F.pil_to_tensor(img).float() / 255.0

        if spec.family == "rotation":
            x2 = _apply_rotation(x, spec.severity)
        elif spec.family == "translation_x":
            x2 = _apply_translation_x(x, int(spec.severity))
        elif spec.family == "translation_y":
            x2 = _apply_translation_y(x, int(spec.severity))
        elif spec.family == "gaussian_noise":
            x2 = _apply_gaussian_noise(x, spec.severity, generator)
        else:
            raise ValueError(f"Unknown corruption family: {spec.family}")

        return preprocess(x2)

    return _transform


def make_clean_transform(preprocess: Callable) -> Callable:
    def _transform(img):
        x = F.pil_to_tensor(img).float() / 255.0
        return preprocess(x)

    return _transform


def group_specs_by_family(specs: Iterable[CorruptionSpec]) -> dict[str, list[CorruptionSpec]]:
    """Utility for plotting grouped by corruption type."""
    out: dict[str, list[CorruptionSpec]] = {}
    for s in specs:
        out.setdefault(s.family, []).append(s)
    for family, family_specs in out.items():
        out[family] = sorted(family_specs, key=lambda v: v.severity)
    return out
