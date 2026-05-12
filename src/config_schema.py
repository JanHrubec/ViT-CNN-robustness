from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _parse_scalar(value: str) -> Any:
    v = value.strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if (v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}")):
        try:
            return yaml.safe_load(v)
        except yaml.YAMLError:
            pass
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_yaml_overrides(payload: dict[str, Any], overrides: list[str]) -> None:
    for raw in overrides:
        if not raw.strip():
            continue
        if "=" not in raw:
            raise ValueError(f"Invalid override (expected KEY=VALUE): {raw!r}")
        key_path, _, value_str = raw.partition("=")
        keys = [k.strip() for k in key_path.split(".") if k.strip()]
        if not keys:
            raise ValueError(f"Invalid key: {raw!r}")
        _set_nested(payload, keys, _parse_scalar(value_str))


@dataclass
class DatasetConfig:
    n_per_class: int = 5
    pool_per_class: int = 50
    cache_dir: str | None = None
    batch_size: int = 128
    num_workers: int = 4


@dataclass
class ModelsConfig:
    names: list[str] = field(default_factory=lambda: ["resnet101", "vit_b_16", "convnext_small"])


@dataclass
class CorruptionsConfig:
    rotation_degrees: list[float] = field(
        default_factory=lambda: [-60, -54, -48, -42, -36, -30, -24, -18, -12, -6, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
    )
    translation_pixels: list[int] = field(
        default_factory=lambda: [-60, -54, -48, -42, -36, -30, -24, -18, -12, -6, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
    )
    gaussian_sigmas: list[float] = field(
        default_factory=lambda: [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50]
    )


@dataclass
class EvaluationConfig:
    device: str = "auto"
    seed: int = 42
    num_repeats: int = 1
    topk: list[int] = field(default_factory=lambda: [1, 5])
    bootstrap_iters: int = 1000
    save_per_sample: bool = True


@dataclass
class MetricsConfig:
    enable_topk: bool = True
    enable_nll: bool = True
    enable_ece: bool = True
    ece_bins: int = 15
    enable_stability: bool = True


@dataclass
class OutputConfig:
    output_dir: str = "./results"
    run_name: str = "base"


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    corruptions: CorruptionsConfig = field(default_factory=CorruptionsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _merge_dataclass(dc_cls: type, values: dict[str, Any] | None):
    if values is None:
        return dc_cls()
    defaults = dc_cls()
    as_dict = defaults.__dict__.copy()
    as_dict.update(values)
    return dc_cls(**as_dict)


def experiment_config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=_merge_dataclass(DatasetConfig, payload.get("dataset")),
        models=_merge_dataclass(ModelsConfig, payload.get("models")),
        corruptions=_merge_dataclass(CorruptionsConfig, payload.get("corruptions")),
        evaluation=_merge_dataclass(EvaluationConfig, payload.get("evaluation")),
        metrics=_merge_dataclass(MetricsConfig, payload.get("metrics")),
        output=_merge_dataclass(OutputConfig, payload.get("output")),
    )


def load_experiment_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = yaml.safe_load(f) or {}
    if overrides:
        apply_yaml_overrides(payload, overrides)
    return experiment_config_from_dict(payload)
