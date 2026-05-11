from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _family_severity_xlabel(family: str) -> str:
    if family == "rotation":
        return "Rotation angle (degrees)"
    if family == "translation_x":
        return "Horizontal translation Δx (pixels)"
    if family == "translation_y":
        return "Vertical translation Δy (pixels)"
    if family == "gaussian_noise":
        return "Gaussian noise σ (additive on [0, 1] RGB channels)"
    return "Corruption severity"


def _family_title_suffix(family: str) -> str:
    mapping = {
        "rotation": "rotation",
        "translation_x": "horizontal translation",
        "translation_y": "vertical translation",
        "gaussian_noise": "Gaussian noise",
    }
    return mapping.get(family, family.replace("_", " "))


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

        plt.figure(figsize=(8, 4.8))
        plotted_any = False

        for model_name in sorted(family_df["model"].unique()):
            mdf = family_df[family_df["model"] == model_name].sort_values("severity")
            if not include_clean:
                mdf = mdf[mdf["split"] == "corrupted"]
            if mdf.empty or metric not in mdf.columns:
                continue

            x = mdf["severity"].astype(float)
            y = mdf[metric].astype(float)
            plt.plot(x, y, marker="o", markersize=4, linewidth=1.8, label=model_name)

            band_col = f"{metric}_std"
            if band_col in mdf.columns:
                band = mdf[band_col].astype(float)
                if band.notna().any():
                    plt.fill_between(x, y - band, y + band, alpha=0.18)
            elif ci_low_col and ci_high_col and ci_low_col in mdf.columns and ci_high_col in mdf.columns:
                lo = mdf[ci_low_col].astype(float)
                hi = mdf[ci_high_col].astype(float)
                if lo.notna().any() and hi.notna().any():
                    plt.fill_between(x, lo, hi, alpha=0.18)

            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        suffix = _family_title_suffix(str(family))
        plt.title(f"{title_prefix} ({suffix})")
        plt.xlabel(_family_severity_xlabel(str(family)))
        plt.ylabel(ylabel)
        plt.grid(alpha=0.28, linestyle="--", linewidth=0.6)
        plt.legend(title="Model", fontsize=9)
        plt.tight_layout()
        safe_family = str(family).replace("/", "_")
        plt.savefig(output / f"{metric}_{safe_family}.png", dpi=200)
        plt.close()


def plot_degradation_curves(results_csv: str | Path, output_dir: str | Path) -> None:
    """Plot degradation metrics per corruption family (aggregated across repeats when std columns exist)."""
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
        ylabel="Top-1 accuracy",
        title_prefix="Top-1 vs corruption",
        ci_low_col="top1_ci_low",
        ci_high_col="top1_ci_high",
    )
    if "top5" in eval_df.columns:
        _plot_family_metric(
            eval_df,
            output,
            metric="top5",
            ylabel="Top-5 accuracy",
            title_prefix="Top-5 vs corruption",
            ci_low_col="top5_ci_low",
            ci_high_col="top5_ci_high",
        )
    if "nll_mean" in eval_df.columns:
        _plot_family_metric(
            eval_df,
            output,
            metric="nll_mean",
            ylabel="Mean NLL (nats)",
            title_prefix="Negative log-likelihood vs corruption",
        )
    if "ece" in eval_df.columns:
        _plot_family_metric(
            eval_df,
            output,
            metric="ece",
            ylabel="ECE",
            title_prefix="Expected calibration error vs corruption",
        )

    if "robustness_ratio_top1" in df.columns:
        rdf = df[df["split"] == "corrupted"].copy()
        if not rdf.empty:
            _plot_family_metric(
                rdf,
                output,
                metric="robustness_ratio_top1",
                ylabel="Robustness ratio (corrupt top-1 / clean top-1)",
                title_prefix="Robustness ratio vs corruption",
            )


def plot_stability_curves(stability_csv: str | Path, output_dir: str | Path) -> None:
    """Plot prediction stability vs corruption severity (aggregated file)."""
    output = Path(output_dir)
    p = Path(stability_csv)
    if not p.is_file():
        return
    df = pd.read_csv(p)
    if df.empty or "corruption_family" not in df.columns:
        return

    for family in sorted(df["corruption_family"].unique()):
        fam_df = df[df["corruption_family"] == family]
        plt.figure(figsize=(8, 4.8))
        for model_name in sorted(fam_df["model"].unique()):
            mdf = fam_df[fam_df["model"] == model_name].sort_values("severity")
            if mdf.empty or "stability_top1" not in mdf.columns:
                continue
            x = mdf["severity"].astype(float)
            y = mdf["stability_top1"].astype(float)
            plt.plot(x, y, marker="o", markersize=4, linewidth=1.8, label=model_name)
            band_col = "stability_top1_std"
            if band_col in mdf.columns:
                band = mdf[band_col].astype(float)
                if band.notna().any():
                    plt.fill_between(x, y - band, y + band, alpha=0.18)

        plt.title(f"Prediction stability vs corruption ({_family_title_suffix(str(family))})")
        plt.xlabel(_family_severity_xlabel(str(family)))
        plt.ylabel("Fraction matching clean top-1 prediction")
        plt.ylim(-0.02, 1.02)
        plt.grid(alpha=0.28, linestyle="--", linewidth=0.6)
        plt.legend(title="Model", fontsize=9)
        plt.tight_layout()
        safe_family = str(family).replace("/", "_")
        plt.savefig(output / f"stability_top1_{safe_family}.png", dpi=200)
        plt.close()


def plot_all_run_artifacts(run_dir: str | Path) -> None:
    """Regenerate plots from final CSVs in a run directory."""
    run_path = Path(run_dir)
    res = run_path / "results.csv"
    if res.is_file():
        plot_degradation_curves(res, run_path)
    stab = run_path / "stability.csv"
    if stab.is_file():
        plot_stability_curves(stab, run_path)
