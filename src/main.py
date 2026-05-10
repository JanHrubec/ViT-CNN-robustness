from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

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
        default="configs/testing_experiment.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--override",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override nested config (dot path), e.g. evaluation.num_repeats=1. Repeat flag for multiple.",
    )
    return parser.parse_args()


def _result_row(result: EvalResult, clean_top1: float | None = None) -> dict:
    row = asdict(result)
    if result.top1 is None or clean_top1 is None:
        row["robustness_ratio_top1"] = None
    else:
        row["robustness_ratio_top1"] = robustness_ratio(clean_top1, result.top1)
    return row


def _aggregate_rows(rows: list[dict], group_cols: list[str]) -> list[dict]:
    if not rows:
        return []

    df = pd.DataFrame(rows)
    if df.empty:
        return []

    numeric_cols = [
        "top1",
        "top5",
        "top1_ci_low",
        "top1_ci_high",
        "top5_ci_low",
        "top5_ci_high",
        "nll_mean",
        "ece",
        "robustness_ratio_top1",
        "sample_count",
    ]
    first_cols = [col for col in ["split", "corruption_family", "corruption_name", "model"] if col in df.columns]

    aggregated: list[dict] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if len(group_cols) == 1:
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        for col in first_cols:
            if col in group.columns:
                row[col] = group.iloc[0][col]
        for col in numeric_cols:
            if col not in group.columns:
                continue
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if values.empty:
                row[col] = None
                row[f"{col}_std"] = None
            else:
                row[col] = float(values.mean())
                row[f"{col}_std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        row["repeat_count"] = int(len(group))
        aggregated.append(row)

    return aggregated


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=list(args.config_overrides or []))

    device = resolve_device(cfg.evaluation.device)

    run_name = f"{cfg.output.run_name}_{timestamp()}"
    run_dir = Path(cfg.output.output_dir) / run_name

    specs = build_corruption_specs(cfg.corruptions)
    topk = tuple(sorted(set(cfg.evaluation.topk)))

    all_repeat_rows: list[dict] = []
    all_repeat_summary_rows: list[dict] = []
    all_per_sample_rows: list[dict] = []

    for model_name in cfg.models.names:
        # Every model bundled with its preprocessing
        bundle = load_pretrained_model(model_name, device)

        repeat_rows: list[dict] = []
        repeat_summary_rows: list[dict] = []
        per_sample_rows: list[dict] = []

        for repeat_index in range(cfg.evaluation.num_repeats):
            repeat_seed = cfg.evaluation.seed + repeat_index
            set_global_seed(repeat_seed)
            base_dataset = build_base_dataset(cfg.dataset, seed=repeat_seed)

            clean_outcome = evaluate_clean(
                model=bundle.model,
                model_name=bundle.name,
                preprocess=bundle.preprocess,
                base_dataset=base_dataset,
                dataset_cfg=cfg.dataset,
                device=device,
                topk=topk,
                bootstrap_iters=cfg.evaluation.bootstrap_iters,
                seed=repeat_seed,
                save_per_sample=cfg.evaluation.save_per_sample,
                metrics_cfg=cfg.metrics,
            )
            clean = clean_outcome.result
            clean_row = _result_row(clean)
            clean_row["repeat"] = repeat_index
            clean_row["seed"] = repeat_seed
            repeat_rows.append(clean_row)
            per_sample_rows.extend([{**row, "repeat": repeat_index, "seed": repeat_seed} for row in clean_outcome.per_sample_rows])

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
                    seed=repeat_seed,
                    bootstrap_iters=cfg.evaluation.bootstrap_iters,
                    save_per_sample=cfg.evaluation.save_per_sample,
                    metrics_cfg=cfg.metrics,
                )
                result = corrupted_outcome.result
                row = _result_row(result, clean_top1=clean.top1)
                row["repeat"] = repeat_index
                row["seed"] = repeat_seed
                repeat_rows.append(row)
                per_sample_rows.extend([{**sample_row, "repeat": repeat_index, "seed": repeat_seed} for sample_row in corrupted_outcome.per_sample_rows])
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

                repeat_summary_rows.append(
                    {
                        "model": bundle.name,
                        "repeat": repeat_index,
                        "seed": repeat_seed,
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

            all_repeat_rows.extend(repeat_rows)
            all_repeat_summary_rows.extend(repeat_summary_rows)
            all_per_sample_rows.extend(per_sample_rows)

    config_snapshot = run_dir / "config_snapshot.json"
    results_csv = run_dir / "results.csv"
    repeat_results_csv = run_dir / "results_repeat.csv"
    summary_csv = run_dir / "summary.csv"
    repeat_summary_csv = run_dir / "summary_repeat.csv"
    per_sample_csv = run_dir / "per_sample.csv"
    stability_repeat_csv = run_dir / "stability_repeat.csv"

    save_json(config_snapshot, asdict(cfg))
    save_csv(repeat_results_csv, all_repeat_rows)
    save_csv(results_csv, _aggregate_rows(all_repeat_rows, ["model", "split", "corruption_family", "corruption_name", "severity"]))
    save_csv(repeat_summary_csv, all_repeat_summary_rows)
    save_csv(summary_csv, _aggregate_rows(all_repeat_summary_rows, ["model", "corruption_family"]))
    if cfg.evaluation.save_per_sample:
        save_csv(per_sample_csv, all_per_sample_rows)
    if cfg.metrics.enable_stability:
        if not cfg.evaluation.save_per_sample:
            raise ValueError("Prediction stability requires save_per_sample=true.")
        stability_rows = compute_prediction_stability(all_per_sample_rows)
        save_csv(stability_repeat_csv, stability_rows)
        save_csv(run_dir / "stability.csv", _aggregate_rows(stability_rows, ["model", "corruption_family", "severity"]))
    plot_degradation_curves(results_csv, run_dir)

    print(f"Done. Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
