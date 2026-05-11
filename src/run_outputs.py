"""Filenames written under each `results/<run_name>_<timestamp>/` directory (self-documenting names)."""

from __future__ import annotations

# JSON
EXPERIMENT_CONFIG_JSON = "experiment_config_used.json"
RUN_OUTPUT_INDEX_JSON = "run_output_index.json"

# Tables (CSV)
EVAL_METRICS_BY_REPEAT_CSV = "eval_metrics_by_repeat.csv"
EVAL_METRICS_CHECKPOINT_AFTER_EACH_MODEL_CSV = "eval_metrics_checkpoint_after_each_model.csv"
EVAL_METRICS_MEAN_AND_STD_OVER_REPEATS_CSV = "eval_metrics_mean_and_std_over_repeats.csv"
EVAL_PER_SAMPLE_PREDICTIONS_AND_METRICS_CSV = "eval_per_sample_predictions_and_metrics.csv"

CORRUPTION_TREND_SUMMARIES_BY_REPEAT_CSV = "corruption_trend_summaries_by_repeat.csv"
CORRUPTION_TREND_SUMMARIES_MEAN_OVER_REPEATS_CSV = "corruption_trend_summaries_mean_over_repeats.csv"

PREDICTION_STABILITY_BY_REPEAT_CSV = "prediction_stability_by_repeat.csv"
PREDICTION_STABILITY_MEAN_AND_STD_OVER_REPEATS_CSV = "prediction_stability_mean_and_std_over_repeats.csv"

# One CSV per model after it finishes (subset of rows for that model only)
PROGRESS_AFTER_MODEL_PREFIX = "eval_metrics_progress_after_model_"


def progress_after_model_csv(model_slug: str) -> str:
    return f"{PROGRESS_AFTER_MODEL_PREFIX}{model_slug}.csv"


PROGRESS_AFTER_MODEL_GLOB = f"{PROGRESS_AFTER_MODEL_PREFIX}*.csv"
