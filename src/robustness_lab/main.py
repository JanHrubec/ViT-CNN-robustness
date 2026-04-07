from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .config_schema import load_experiment_config
from .corruptions import build_corruption_specs
from .datasets import build_base_dataset
from .io_utils import save_csv, save_json, timestamp
from .metrics import audc, endpoint_delta, linear_trend_slope, robustness_ratio, compute_prediction_stability
from .models import load_pretrained_model
from .plots import plot_degradation_curves
from .runner import EvalResult, evaluate_clean, evaluate_corruption
from .utils import resolve_device, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def _result_row(result: EvalResult, clean_top1: float | None = None) -> dict:
    row = asdict(result)
    if result.top1 is None or clean_top1 is None:
        row["robustness_ratio_top1"] = None
    else:
        row["robustness_ratio_top1"] = robustness_ratio(clean_top1, result.top1)
    return row


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)

    # One seed for reproducibility
    set_global_seed(cfg.evaluation.seed)
    device = resolve_device(cfg.evaluation.device)

    run_name = f"{cfg.output.run_name}_{timestamp()}"
    run_dir = Path(cfg.output.output_dir) / run_name

    base_dataset = build_base_dataset(cfg.dataset, seed=cfg.evaluation.seed)
    specs = build_corruption_specs(cfg.corruptions)
    topk = tuple(sorted(set(cfg.evaluation.topk)))

    rows: list[dict] = []
    summary_rows: list[dict] = []
    per_sample_rows: list[dict] = []

    for model_name in cfg.models.names:
        # Every model bundled with its preprocessing
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
            metrics_cfg=cfg.metrics,
        )
        clean = clean_outcome.result
        rows.append(_result_row(clean))
        per_sample_rows.extend(clean_outcome.per_sample_rows)

        family_top1: dict[str, list[tuple[float, float]]] = {}
        family_nll: dict[str, list[tuple[float, float]]] = {}
        family_ece: dict[str, list[tuple[float, float]]] = {}

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
                metrics_cfg=cfg.metrics,
            )
            result = corrupted_outcome.result
            rows.append(_result_row(result, clean_top1=clean.top1))
            per_sample_rows.extend(corrupted_outcome.per_sample_rows)
            if result.top1 is not None:
                family_top1.setdefault(spec.family, []).append((spec.severity, result.top1))
            if result.nll_mean is not None:
                family_nll.setdefault(spec.family, []).append((spec.severity, result.nll_mean))
            if result.ece is not None:
                family_ece.setdefault(spec.family, []).append((spec.severity, result.ece))

        all_families = sorted(set(family_top1) | set(family_nll) | set(family_ece))
        for family in all_families:
            top1_points = sorted(family_top1.get(family, []), key=lambda x: x[0])
            nll_points = sorted(family_nll.get(family, []), key=lambda x: x[0])
            ece_points = sorted(family_ece.get(family, []), key=lambda x: x[0])

            top1_sev = [p[0] for p in top1_points]
            top1_vals = [p[1] for p in top1_points]
            nll_sev = [p[0] for p in nll_points]
            nll_vals = [p[1] for p in nll_points]
            ece_sev = [p[0] for p in ece_points]
            ece_vals = [p[1] for p in ece_points]

            summary_rows.append(
                {
                    "model": bundle.name,
                    "corruption_family": family,
                    "clean_top1": clean.top1,
                    "clean_nll": clean.nll_mean,
                    "clean_ece": clean.ece,
                    "audc_top1": audc(top1_sev, top1_vals) if (clean.top1 is not None and len(top1_vals) > 1) else None,
                    "audc_nll": audc(nll_sev, nll_vals) if len(nll_vals) > 1 else None,
                    "audc_ece": audc(ece_sev, ece_vals) if len(ece_vals) > 1 else None,
                    "slope_top1": linear_trend_slope(top1_sev, top1_vals) if len(top1_vals) > 1 else None,
                    "slope_nll": linear_trend_slope(nll_sev, nll_vals) if len(nll_vals) > 1 else None,
                    "slope_ece": linear_trend_slope(ece_sev, ece_vals) if len(ece_vals) > 1 else None,
                    "delta_top1_max": endpoint_delta(clean.top1, top1_vals[-1]) if (clean.top1 is not None and top1_vals) else None,
                    "delta_nll_max": endpoint_delta(clean.nll_mean, nll_vals[-1]) if (clean.nll_mean is not None and nll_vals) else None,
                    "delta_ece_max": endpoint_delta(clean.ece, ece_vals[-1]) if (clean.ece is not None and ece_vals) else None,
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
    if cfg.metrics.enable_stability:
        if not cfg.evaluation.save_per_sample:
            raise ValueError("Prediction stability requires save_per_sample=true.")
        stability_rows = compute_prediction_stability(per_sample_rows)
        save_csv(run_dir / "stability.csv", stability_rows)
    plot_degradation_curves(results_csv, run_dir)

    print(f"Done. Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
