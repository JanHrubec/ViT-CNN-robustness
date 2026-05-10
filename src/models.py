from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import timm
import timm.data
import torch

# Map config / legacy names to timm checkpoints (ImageNet-1k only).
TIMM_MODEL_ALIASES: dict[str, str] = {
    "resnet101": "resnet101.a1_in1k",
    "resnet101_in1k": "resnet101.a1_in1k",
    "vit_b_16": "vit_base_patch16_224.augreg_in1k",
    "vit_b16_in1k": "vit_base_patch16_224.augreg_in1k",
    "convnext_small": "convnext_small.fb_in1k",
    "convnext_small_in1k": "convnext_small.fb_in1k",
}


@dataclass
class ModelBundle:
    """Model plus inference-time preprocessing from the same timm checkpoint."""

    name: str
    timm_id: str
    model: torch.nn.Module
    preprocess: Callable
    class_names: list[str]


def resolve_timm_model_id(model_name: str) -> str:
    """
    Resolve a config model name to a timm model id.

    - If `model_name` matches a known alias, use the alias mapping (defaults to ImageNet-1k-only checkpoints).
    - Otherwise, treat `model_name` as an explicit timm identifier and pass it through.
    """
    raw = model_name.strip()
    key = raw.lower()
    return TIMM_MODEL_ALIASES.get(key, raw)


def load_pretrained_model(model_name: str, device: torch.device) -> ModelBundle:
    """Load a timm model (IN-1k only weights) and its standard eval transform."""
    timm_name = resolve_timm_model_id(model_name)
    model = timm.create_model(timm_name, pretrained=True)
    model.eval().to(device)

    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)

    return ModelBundle(
        name=model_name.strip(),
        timm_id=timm_name,
        model=model,
        preprocess=preprocess,
        class_names=[],
    )
