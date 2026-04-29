from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torchvision.models import (
    ResNet101_Weights,
    ViT_B_16_Weights,
    ConvNeXt_Small_Weights,
    resnet101,
    vit_b_16,
    convnext_small,
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

    if model_name == "resnet101":
        weights = ResNet101_Weights.IMAGENET1K_V1
        model = resnet101(weights=weights)
    elif model_name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
    elif model_name == "convnext_small":
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        model = convnext_small(weights=weights)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            "Choose from: resnet101, vit_b_16, convnext_small."
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
