from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from .config_schema import CorruptionsConfig


# Approximate ImageNet mean in 0–255 space — rotation / translation borders (before normalisation).
_IN_MEAN_FILL_255 = [123, 116, 103]
_IN_MEAN_FILL_01 = [123 / 255.0, 116 / 255.0, 103 / 255.0]


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


def _tensor01_to_pil(x: torch.Tensor) -> Image.Image:
    x = torch.clamp(x, 0.0, 1.0)
    return F.to_pil_image(x)


def _pil_to_tensor01(pil_image: Image.Image) -> torch.Tensor:
    return F.pil_to_tensor(pil_image.convert("RGB")).float() / 255.0


def _apply_rotation_pil(pil_image: Image.Image, deg: float) -> Image.Image:
    return F.rotate(
        pil_image,
        angle=deg,
        interpolation=InterpolationMode.BILINEAR,
        fill=_IN_MEAN_FILL_255,
    )


def _apply_translation_x_pil(pil_image: Image.Image, px: int) -> Image.Image:
    x = _pil_to_tensor01(pil_image)
    x2 = F.affine(
        x,
        angle=0.0,
        translate=[float(px), 0.0],
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=_IN_MEAN_FILL_01,
    )
    return _tensor01_to_pil(x2)


def _apply_translation_y_pil(pil_image: Image.Image, px: int) -> Image.Image:
    x = _pil_to_tensor01(pil_image)
    x2 = F.affine(
        x,
        angle=0.0,
        translate=[0.0, float(px)],
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=_IN_MEAN_FILL_01,
    )
    return _tensor01_to_pil(x2)


def _apply_gaussian_noise_pil(pil_image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    if sigma <= 0:
        return pil_image
    arr = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    noise = rng.normal(0.0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def apply_corruption_spec_pil(spec: CorruptionSpec, pil_image: Image.Image, seed: int) -> Image.Image:
    """Apply one corruption in PIL space (no model normalisation). Used for previews and shared with transforms."""
    rng = np.random.default_rng(seed)
    if spec.family == "rotation":
        return _apply_rotation_pil(pil_image, spec.severity)
    if spec.family == "translation_x":
        return _apply_translation_x_pil(pil_image, int(spec.severity))
    if spec.family == "translation_y":
        return _apply_translation_y_pil(pil_image, int(spec.severity))
    if spec.family == "gaussian_noise":
        return _apply_gaussian_noise_pil(pil_image, float(spec.severity), rng)
    raise ValueError(f"Unknown corruption family: {spec.family}")


def save_corruption_preview_gallery(out_dir: str | Path, pil_image: Image.Image, specs: list[CorruptionSpec], seed: int) -> None:
    """Save the reference image and each corrupted variant (PIL after corruption, before timm preprocess)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = pil_image.convert("RGB")
    ref.save(out / "00_reference_image__clean_no_corruption.png")
    for spec in specs:
        corrupted = apply_corruption_spec_pil(spec, ref.copy(), seed=seed)
        safe_name = spec.name.replace("/", "_")
        corrupted.save(out / f"corrupted_image__{safe_name}.png")


def make_corruption_transform(
    spec: CorruptionSpec,
    preprocess: Callable,
    seed: int,
) -> Callable:
    """
    PIL image → corruption (still PIL) → model preprocess → tensor.

    Gaussian noise uses additive noise in [0, 1] image space before normalisation.
    The RNG advances across samples so each image gets a fresh noise draw.
    """
    gaussian_rng = np.random.default_rng(seed) if spec.family == "gaussian_noise" else None

    def _transform(img) -> torch.Tensor:
        if spec.family == "gaussian_noise" and gaussian_rng is not None:
            corrupted = _apply_gaussian_noise_pil(img, float(spec.severity), gaussian_rng)
        else:
            corrupted = apply_corruption_spec_pil(spec, img, seed)
        return preprocess(corrupted)

    return _transform


def make_clean_transform(preprocess: Callable) -> Callable:
    def _transform(img):
        return preprocess(img)

    return _transform


def group_specs_by_family(specs: Iterable[CorruptionSpec]) -> dict[str, list[CorruptionSpec]]:
    """Utility for plotting grouped by corruption type."""
    out: dict[str, list[CorruptionSpec]] = {}
    for s in specs:
        out.setdefault(s.family, []).append(s)
    for family, family_specs in out.items():
        out[family] = sorted(family_specs, key=lambda v: v.severity)
    return out
