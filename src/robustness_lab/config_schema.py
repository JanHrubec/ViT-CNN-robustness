from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    # `imagenet_folder` assumes ImageFolder-style directory structure.
    name: str = "imagenet_folder"
    root: str = "./data/imagenet"
    split: str = "val"
    subset_per_class: int | None = 10
    batch_size: int = 64
    num_workers: int = 4


@dataclass
class ModelsConfig:
    # Start with classic baseline pair; add ConvNeXt as optional extension.
    names: list[str] = field(default_factory=lambda: ["resnet50", "vit_b_16"])


@dataclass
class CorruptionsConfig:
    # Severity sweeps are explicit lists, so every run is easy to reproduce.
    rotation_degrees: list[float] = field(default_factory=lambda: [0, 5, 10, 15, 20, 30])
    translation_pixels: list[int] = field(default_factory=lambda: [0, 4, 8, 12, 16])
    gaussian_sigmas: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.10, 0.15])


@dataclass
class EvaluationConfig:
    device: str = "auto"
    seed: int = 42
    topk: list[int] = field(default_factory=lambda: [1, 5])
    bootstrap_iters: int = 1000
    save_per_sample: bool = True


@dataclass
class OutputConfig:
    output_dir: str = "./results"
    run_name: str = "baseline"


@dataclass
class ExperimentConfig:
    # Top-level container mirrors YAML sections one-to-one.
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    corruptions: CorruptionsConfig = field(default_factory=CorruptionsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _merge_dataclass(dc_cls: type, values: dict[str, Any] | None):
    """Merge user-provided values over dataclass defaults."""
    if values is None:
        return dc_cls()
    defaults = dc_cls()
    as_dict = defaults.__dict__.copy()
    as_dict.update(values)
    return dc_cls(**as_dict)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load YAML config and coerce it into typed config dataclasses."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    return ExperimentConfig(
        dataset=_merge_dataclass(DatasetConfig, payload.get("dataset")),
        models=_merge_dataclass(ModelsConfig, payload.get("models")),
        corruptions=_merge_dataclass(CorruptionsConfig, payload.get("corruptions")),
        evaluation=_merge_dataclass(EvaluationConfig, payload.get("evaluation")),
        output=_merge_dataclass(OutputConfig, payload.get("output")),
    )
