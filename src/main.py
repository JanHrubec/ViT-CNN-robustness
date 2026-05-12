from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config_schema import load_experiment_config
from .corruptions import build_corruption_specs, save_corruption_preview_gallery
from .datasets import build_base_dataset
from .io_utils import append_csv_rows, save_csv, save_json, timestamp
from . import run_outputs as rout
from .metrics import (
    PredictionStabilityAggregator,
    audc,
    endpoint_delta,
    linear_trend_slope,
    robustness_ratio,
)
from .models import load_pretrained_model
from .plots import plot_all_run_artifacts
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
    parser.add_argument(
        "--override",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override nested config (dot path), e.g. evaluation.num_repeats=1. Repeat flag for multiple.",
    )
    return parser.parse_args()


def result_row(result: EvalResult, clean_top1: float | None = None) -> dict:
    row = asdict(result)
    if result.top1 is None or clean_top1 is None:
        row["robustness_ratio_top1"] = None
    else:
        row["robustness_ratio_top1"] = robustness_ratio(clean_top1, result.top1)
    return row


def aggregate_rows(rows: list[dict], group_cols: list[str]) -> list[dict]:
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
        "stability_top1",
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


def unique_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "model"


def csv_data_rows(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def write_metrics(run_dir: Path) -> None:
    files = [
        rout.EXPERIMENT_CONFIG_JSON,
        rout.EVAL_METRICS_BY_REPEAT_CSV,
        rout.EVAL_METRICS_CHECKPOINT_AFTER_EACH_MODEL_CSV,
        rout.EVAL_METRICS_MEAN_AND_STD_OVER_REPEATS_CSV,
        rout.EVAL_PER_SAMPLE_PREDICTIONS_AND_METRICS_CSV,
        rout.CORRUPTION_TREND_SUMMARIES_BY_REPEAT_CSV,
        rout.CORRUPTION_TREND_SUMMARIES_MEAN_OVER_REPEATS_CSV,
        rout.PREDICTION_STABILITY_BY_REPEAT_CSV,
        rout.PREDICTION_STABILITY_MEAN_AND_STD_OVER_REPEATS_CSV,
    ]
    payload: dict[str, object] = {"artifacts": {}, "plots": [], "progress_csv": []}
    for name in files:
        p = run_dir / name
        payload["artifacts"][name] = {"exists": p.is_file(), "data_rows": csv_data_rows(p)}
    payload["plots"] = sorted(p.name for p in run_dir.glob("plot_*.png"))
    payload["progress_csv"] = sorted(p.name for p in run_dir.glob(rout.PROGRESS_AFTER_MODEL_GLOB))
    manifest = run_dir / rout.RUN_OUTPUT_INDEX_JSON
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def refresh_intermediate_aggregates(run_dir: Path) -> None:
    repeat_path = run_dir / rout.EVAL_METRICS_BY_REPEAT_CSV
    if not repeat_path.is_file() or repeat_path.stat().st_size == 0:
        return
    df = pd.read_csv(repeat_path)
    rows = df.to_dict("records")
    save_csv(
        run_dir / rout.EVAL_METRICS_CHECKPOINT_AFTER_EACH_MODEL_CSV,
        aggregate_rows(rows, ["model", "split", "corruption_family", "corruption_name", "severity"]),
    )


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=list(args.config_overrides or []))

    device = resolve_device(cfg.evaluation.device)

    run_name = f"{cfg.output.run_name}_{timestamp()}"
    run_dir = Path(cfg.output.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    repeat_results_csv = run_dir / rout.EVAL_METRICS_BY_REPEAT_CSV
    per_sample_csv = run_dir / rout.EVAL_PER_SAMPLE_PREDICTIONS_AND_METRICS_CSV
    repeat_summary_csv = run_dir / rout.CORRUPTION_TREND_SUMMARIES_BY_REPEAT_CSV
    stability_repeat_csv = run_dir / rout.PREDICTION_STABILITY_BY_REPEAT_CSV
    results_csv = run_dir / rout.EVAL_METRICS_MEAN_AND_STD_OVER_REPEATS_CSV
    summary_csv = run_dir / rout.CORRUPTION_TREND_SUMMARIES_MEAN_OVER_REPEATS_CSV
    stability_csv = run_dir / rout.PREDICTION_STABILITY_MEAN_AND_STD_OVER_REPEATS_CSV

    save_json(run_dir / rout.EXPERIMENT_CONFIG_JSON, asdict(cfg))

    specs = build_corruption_specs(cfg.corruptions)
    topk = tuple(sorted(set(cfg.evaluation.topk)))

    stability_agg = PredictionStabilityAggregator() if cfg.metrics.enable_stability else None

    for model_index, model_name in enumerate(cfg.models.names):
        bundle = load_pretrained_model(model_name, device)

        for repeat_index in range(cfg.evaluation.num_repeats):
            repeat_seed = cfg.evaluation.seed + repeat_index
            set_global_seed(repeat_seed)
            base_dataset = build_base_dataset(cfg.dataset, seed=repeat_seed)

            if model_index == 0:
                ref_img, _ = base_dataset[0]
                preview_dir = run_dir / "corruption_previews" / f"repeat_{repeat_index}_seed_{repeat_seed}"
                save_corruption_preview_gallery(preview_dir, ref_img, specs, repeat_seed)

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
            clean_row = result_row(clean)
            clean_row["repeat"] = repeat_index
            clean_row["seed"] = repeat_seed
            append_csv_rows(repeat_results_csv, [clean_row])

            ps_clean = [{**row, "repeat": repeat_index, "seed": repeat_seed} for row in clean_outcome.per_sample_rows]
            if cfg.evaluation.save_per_sample and ps_clean:
                append_csv_rows(per_sample_csv, ps_clean)

            if stability_agg is not None:
                stability_agg.ingest_clean_rows(bundle.name, repeat_index, ps_clean)

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
                row = result_row(result, clean_top1=clean.top1)
                row["repeat"] = repeat_index
                row["seed"] = repeat_seed
                append_csv_rows(repeat_results_csv, [row])

                ps_corrupt = [
                    {**sample_row, "repeat": repeat_index, "seed": repeat_seed}
                    for sample_row in corrupted_outcome.per_sample_rows
                ]
                if cfg.evaluation.save_per_sample and ps_corrupt:
                    append_csv_rows(per_sample_csv, ps_corrupt)

                if stability_agg is not None:
                    stability_agg.ingest_corrupted_rows(
                        bundle.name,
                        repeat_index,
                        spec.family,
                        float(spec.severity),
                        ps_corrupt,
                    )
                    stab_row = stability_agg.row_for_condition(
                        bundle.name, repeat_index, spec.family, float(spec.severity)
                    )
                    if stab_row is not None:
                        append_csv_rows(stability_repeat_csv, [stab_row])

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

                summary_row = {
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
                    "delta_top1_max": endpoint_delta(clean.top1, top1_vals[-1])
                    if (clean.top1 is not None and top1_vals)
                    else None,
                    "delta_nll_max": endpoint_delta(clean.nll_mean, nll_vals[-1])
                    if (clean.nll_mean is not None and nll_vals)
                    else None,
                    "delta_ece_max": endpoint_delta(clean.ece, ece_vals[-1]) if (clean.ece is not None and ece_vals) else None,
                }
                append_csv_rows(repeat_summary_csv, [summary_row])

        refresh_intermediate_aggregates(run_dir)
        slug = unique_slug(bundle.name)
        if repeat_results_csv.is_file() and repeat_results_csv.stat().st_size > 0:
            df_all = pd.read_csv(repeat_results_csv)
            df_model = df_all[df_all["model"] == bundle.name]
            save_csv(
                run_dir / rout.progress_after_model_csv(slug),
                aggregate_rows(df_model.to_dict("records"), ["model", "split", "corruption_family", "corruption_name", "severity"]),
            )

    if repeat_results_csv.is_file() and repeat_results_csv.stat().st_size > 0:
        df_repeat = pd.read_csv(repeat_results_csv)
        repeat_rows = df_repeat.to_dict("records")
        save_csv(results_csv, aggregate_rows(repeat_rows, ["model", "split", "corruption_family", "corruption_name", "severity"]))

    if repeat_summary_csv.is_file() and repeat_summary_csv.stat().st_size > 0:
        df_sum_rep = pd.read_csv(repeat_summary_csv)
        save_csv(summary_csv, aggregate_rows(df_sum_rep.to_dict("records"), ["model", "corruption_family"]))
    if cfg.metrics.enable_stability:
        if not cfg.evaluation.save_per_sample:
            raise ValueError("Prediction stability requires save_per_sample=true.")
        if stability_repeat_csv.is_file() and stability_repeat_csv.stat().st_size > 0:
            df_stab_rep = pd.read_csv(stability_repeat_csv)
            save_csv(stability_csv, aggregate_rows(df_stab_rep.to_dict("records"), ["model", "corruption_family", "severity"]))
        elif stability_agg is not None:
            save_csv(stability_csv, aggregate_rows(stability_agg.to_rows(), ["model", "corruption_family", "severity"]))

    plot_all_run_artifacts(run_dir)
    write_metrics(run_dir)

    print(f"Done. Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
