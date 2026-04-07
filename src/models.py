from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torchvision.models import (
    ResNet18_Weights,
    ViT_B_32_Weights,
    ConvNeXt_Tiny_Weights,
    resnet18,
    vit_b_32,
    convnext_tiny,
)


@dataclass
class ModelBundle:
    """Bundle model and metadata"""
    name: str
    model: torch.nn.Module
    preprocess: Callable
    class_names: list[str]


def load_pretrained_model(model_name: str, device: torch.device) -> ModelBundle:
    """Load pretrained torchvision model with transforms"""
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    elif model_name == "vit_b_32":
        weights = ViT_B_32_Weights.IMAGENET1K_V1
        model = vit_b_32(weights=weights)
    elif model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = convnext_tiny(weights=weights)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            "Choose from: resnet18, vit_b_32."
        )

    # pure inference
    model.eval().to(device)

    # Categories for class-level statistics?
    categories = weights.meta.get("categories", [])
    class_names = [str(x) for x in categories] if categories else []

    return ModelBundle(
        name=model_name,
        model=model,
        preprocess=weights.transforms(),
        class_names=class_names,
    )
