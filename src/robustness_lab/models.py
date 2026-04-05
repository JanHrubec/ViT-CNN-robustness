from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torchvision.models import (
    ResNet18_Weights,
    ViT_B_32_Weights,
    resnet18,
    vit_b_32,
)


@dataclass
class ModelBundle:
    """Bundle model and metadata needed during evaluation."""
    name: str
    model: torch.nn.Module
    preprocess: Callable
    class_names: list[str]


def load_pretrained_model(model_name: str, device: torch.device) -> ModelBundle:
    """Load a supported pretrained torchvision model with matching transforms."""
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    elif model_name == "vit_b_32":
        weights = ViT_B_32_Weights.IMAGENET1K_V1
        model = vit_b_32(weights=weights)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            "Choose from: resnet18, vit_b_32."
        )

    # `eval()` disables dropout-like layers; this is pure inference.
    model.eval().to(device)

    # Categories are useful later if you want class-level diagnostics.
    categories = weights.meta.get("categories", [])
    class_names = [str(x) for x in categories] if categories else []

    return ModelBundle(
        name=model_name,
        model=model,
        preprocess=weights.transforms(),
        class_names=class_names,
    )
