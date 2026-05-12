from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from src.config_schema import load_experiment_config
from src.corruptions import apply_corruption_spec_pil, build_corruption_specs, group_specs_by_family


def default_thumbnail_image(size: int = 224) -> Image.Image:
    px = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(px)
    for y in range(size):
        for x in range(size):
            r = int(255 * x / max(size - 1, 1))
            g = int(255 * y / max(size - 1, 1))
            b = int(255 * (x + y) / max(2 * (size - 1), 1))
            d.point((x, y), fill=(r, g, b))
    return px


def load_font(size: int = 11):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def build_montage(
    *,
    ref: Image.Image,
    specs_by_family: dict[str, list],
    thumb: int,
    pad: int,
    label_h: int,
    seed: int,
) -> Image.Image:
    """Stack families vertically; within each family, severities left→right."""
    order = ("rotation", "translation_x", "translation_y", "gaussian_noise")
    font = load_font(11)
    rows: list[Image.Image] = []

    for family in order:
        fam_specs = specs_by_family.get(family, [])
        if not fam_specs:
            continue
        fam_specs = sorted(fam_specs, key=lambda s: s.severity)
        n = len(fam_specs)
        row_w = pad + n * (thumb + pad)
        row_h = pad + label_h + thumb + pad
        row = Image.new("RGB", (row_w, row_h), (32, 32, 32))
        dr = ImageDraw.Draw(row)
        dr.text((pad, pad), f"{family} ({n} levels)", fill=(240, 240, 240), font=font)

        for i, spec in enumerate(fam_specs):
            x0 = pad + i * (thumb + pad)
            y0 = pad + label_h
            corrupted = apply_corruption_spec_pil(spec, ref.copy(), seed + i * 9973)
            corrupted = corrupted.convert("RGB").resize((thumb, thumb), Image.Resampling.BILINEAR)
            row.paste(corrupted, (x0, y0))
            label = spec.name.replace("_", " ")
            dr.text((x0, y0 + thumb + 2), label[:18], fill=(200, 200, 200), font=font)

        rows.append(row)

    if not rows:
        raise RuntimeError("No corruption specs found in config.")

    w = max(r.width for r in rows)
    h = sum(r.height for r in rows)
    out = Image.new("RGB", (w, h), (24, 24, 24))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Render one montage of all corruption levels from a benchmark YAML.")
    p.add_argument("--config", type=str, default="configs/base.yaml", help="Experiment YAML (uses corruptions block).")
    p.add_argument("--image", type=str, default="", help="Reference RGB image; default = synthetic gradient.")
    p.add_argument("--out", type=str, default="corruption_montage.png", help="Output PNG path.")
    p.add_argument("--thumb", type=int, default=120, help="Thumbnail edge length (pixels).")
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed (Gaussian noise varies per column).")
    args = p.parse_args()

    cfg = load_experiment_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    specs = build_corruption_specs(cfg.corruptions)
    grouped = group_specs_by_family(specs)

    if args.image:
        ref = Image.open(args.image).convert("RGB")
    else:
        ref = default_thumbnail_image(224)

    montage = build_montage(
        ref=ref,
        specs_by_family=grouped,
        thumb=max(48, args.thumb),
        pad=4,
        label_h=22,
        seed=args.seed,
    )
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_path)
    print(f"Wrote {out_path} ({montage.size[0]}×{montage.size[1]} px)")


if __name__ == "__main__":
    main()
