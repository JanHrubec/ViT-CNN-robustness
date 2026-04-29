from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_family_metric(
    eval_df: pd.DataFrame,
    output: Path,
    metric: str,
    ylabel: str,
    title_prefix: str,
    ci_low_col: str | None = None,
    ci_high_col: str | None = None,
    include_clean: bool = False,
) -> None:
    for family in sorted(eval_df["corruption_family"].unique()):
        family_df = eval_df[eval_df["corruption_family"] == family]

        plt.figure(figsize=(7, 4.5))
        plotted_any = False

        for model_name in sorted(family_df["model"].unique()):
            mdf = family_df[family_df["model"] == model_name].sort_values("severity")
            if not include_clean:
                mdf = mdf[mdf["split"] == "corrupted"]
            if mdf.empty or metric not in mdf.columns:
                continue

            x = mdf["severity"].astype(float)
            y = mdf[metric].astype(float)
            plt.plot(x, y, marker="o", label=model_name)

            band_col = f"{metric}_std"
            if band_col in mdf.columns:
                band = mdf[band_col].astype(float)
                if band.notna().any():
                    plt.fill_between(x, y - band, y + band, alpha=0.15)
            elif ci_low_col and ci_high_col and ci_low_col in mdf.columns and ci_high_col in mdf.columns:
                lo = mdf[ci_low_col].astype(float)
                hi = mdf[ci_high_col].astype(float)
                if lo.notna().any() and hi.notna().any():
                    plt.fill_between(x, lo, hi, alpha=0.15)

            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.title(f"{title_prefix}: {family}")
        plt.xlabel("Severity")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"{metric}_{family}.png", dpi=180)
        plt.close()


def plot_degradation_curves(results_csv: str | Path, output_dir: str | Path) -> None:
    """Plot the core degradation metrics per corruption family."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    if df.empty:
        return

    required = {"model", "corruption_family", "severity", "top1", "split"}
    if not required.issubset(set(df.columns)):
        return

    eval_df = df[df["split"] == "corrupted"].copy()
    if eval_df.empty:
        return

    _plot_family_metric(
        eval_df,
        output,
        metric="top1",
        ylabel="Top-1 Accuracy",
        title_prefix="Robustness curve",
        ci_low_col="top1_ci_low",
        ci_high_col="top1_ci_high",
    )
    if "top5" in eval_df.columns:
        _plot_family_metric(
            eval_df,
            output,
            metric="top5",
            ylabel="Top-5 Accuracy",
            title_prefix="Top-5 robustness curve",
            ci_low_col="top5_ci_low",
            ci_high_col="top5_ci_high",
        )
    if "nll_mean" in eval_df.columns:
        _plot_family_metric(
            eval_df,
            output,
            metric="nll_mean",
            ylabel="Negative Log-Likelihood",
            title_prefix="NLL curve",
        )
    if "ece" in eval_df.columns:
        _plot_family_metric(
            eval_df,
            output,
            metric="ece",
            ylabel="Expected Calibration Error",
            title_prefix="ECE curve",
        )
    if "robustness_ratio_top1" in df.columns:
        _plot_family_metric(
            df,
            output,
            metric="robustness_ratio_top1",
            ylabel="Robustness ratio (corrupt / clean)",
            title_prefix="Robustness ratio curve",
        )
