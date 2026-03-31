from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .config_schema import load_experiment_config
from .corruptions import build_corruption_specs
from .datasets import build_base_dataset
from .io_utils import save_csv, save_json, utc_timestamp
from .metrics import audc, robustness_ratio
from .models import load_pretrained_model
from .plots import plot_degradation_curves
from .runner import EvalResult, evaluate_clean, evaluate_corruption
from .utils import ensure_dir, resolve_device, set_global_seed


def parse_args() -> argparse.Namespace:
    # Keep CLI simple: one config file drives the whole run.
    parser = argparse.ArgumentParser(description="Run CNN vs ViT robustness benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_experiment.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def _result_row(result: EvalResult, clean_top1: float | None = None) -> dict:
    # Convert dataclass to a plain dict so CSV writing is straightforward.
    row = asdict(result)
    if clean_top1 is None:
        # Clean baseline should always have robustness ratio 1.0.
        row["robustness_ratio_top1"] = 1.0 if result.split == "clean" else 0.0
    else:
        # For corrupted rows, ratio is relative to this model's clean performance.
        row["robustness_ratio_top1"] = robustness_ratio(clean_top1, result.top1)
    return row


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)

    # One seed for reproducibility across sampling and noise generation.
    set_global_seed(cfg.evaluation.seed)
    device = resolve_device(cfg.evaluation.device)

    run_name = f"{cfg.output.run_name}_{utc_timestamp()}"
    run_dir = ensure_dir(Path(cfg.output.output_dir) / run_name)

    base_dataset = build_base_dataset(cfg.dataset, seed=cfg.evaluation.seed)
    specs = build_corruption_specs(cfg.corruptions)
    topk = tuple(sorted(set(cfg.evaluation.topk)))

    rows: list[dict] = []
    summary_rows: list[dict] = []
    per_sample_rows: list[dict] = []

    for model_name in cfg.models.names:
        # Every model comes bundled with its own expected preprocessing.
        bundle = load_pretrained_model(model_name, device)

        clean_outcome = evaluate_clean(
            model=bundle.model,
            model_name=bundle.name,
            preprocess=bundle.preprocess,
            base_dataset=base_dataset,
            dataset_cfg=cfg.dataset,
            device=device,
            topk=topk,
            bootstrap_iters=cfg.evaluation.bootstrap_iters,
            seed=cfg.evaluation.seed,
            save_per_sample=cfg.evaluation.save_per_sample,
        )
        clean = clean_outcome.result
        rows.append(_result_row(clean))
        per_sample_rows.extend(clean_outcome.per_sample_rows)

        family_scores: dict[str, list[tuple[float, float]]] = {}

        for spec in specs:
            corrupted_outcome = evaluate_corruption(
                model=bundle.model,
                model_name=bundle.name,
                preprocess=bundle.preprocess,
                base_dataset=base_dataset,
                dataset_cfg=cfg.dataset,
                device=device,
                topk=topk,
                spec=spec,
                seed=cfg.evaluation.seed,
                bootstrap_iters=cfg.evaluation.bootstrap_iters,
                save_per_sample=cfg.evaluation.save_per_sample,
            )
            result = corrupted_outcome.result
            rows.append(_result_row(result, clean_top1=clean.top1))
            per_sample_rows.extend(corrupted_outcome.per_sample_rows)
            family_scores.setdefault(spec.family, []).append((spec.severity, result.top1))

        # Summarize each corruption family into one number for easier comparison.
        for family, points in family_scores.items():
            points = sorted(points, key=lambda x: x[0])
            severities = [p[0] for p in points]
            accuracies = [p[1] for p in points]
            summary_rows.append(
                {
                    "model": bundle.name,
                    "corruption_family": family,
                    "clean_top1": clean.top1,
                    "audc_top1": audc(severities, accuracies),
                }
            )

    config_snapshot = run_dir / "config_snapshot.json"
    results_csv = run_dir / "results.csv"
    summary_csv = run_dir / "summary.csv"
    per_sample_csv = run_dir / "per_sample.csv"

    save_json(config_snapshot, asdict(cfg))
    save_csv(results_csv, rows)
    save_csv(summary_csv, summary_rows)
    if cfg.evaluation.save_per_sample:
        save_csv(per_sample_csv, per_sample_rows)
    # Generate quick diagnostic figures right after metrics are saved.
    plot_degradation_curves(results_csv, run_dir)

    print(f"Done. Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
