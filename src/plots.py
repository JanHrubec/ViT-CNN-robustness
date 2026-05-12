from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import run_outputs as out


def corruption_family_name(family: str) -> str:
    return {
        "rotation": "Rotation",
        "translation_x": "Horizontal translation (Δx)",
        "translation_y": "Vertical translation (Δy)",
        "gaussian_noise": "Gaussian noise",
    }.get(family, family.replace("_", " ").title())


def family_severity_xlabel(family: str) -> str:
    if family == "rotation":
        return "Corruption: rotation angle (degrees)"
    if family == "translation_x":
        return "Corruption: horizontal shift (pixels, positive = right)"
    if family == "translation_y":
        return "Corruption: vertical shift (pixels, positive = down)"
    if family == "gaussian_noise":
        return "Corruption: Gaussian noise σ (additive on [0, 1] RGB before normalisation)"
    return "Corruption severity"


def plot_family_metric(eval_df: pd.DataFrame, output: Path, *, file_stem: str, metric: str, ylabel: str, title_main: str, subtitle_dataset: str, run_folder: str, ci_low_col: str | None = None, ci_high_col: str | None = None, include_clean: bool = False) -> None:
    for family in sorted(eval_df["corruption_family"].unique()):
        family_df = eval_df[eval_df["corruption_family"] == family]

        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        plotted_any = False

        for model_name in sorted(family_df["model"].unique()):
            mdf = family_df[family_df["model"] == model_name].sort_values("severity")
            if not include_clean:
                mdf = mdf[mdf["split"] == "corrupted"]
            if mdf.empty or metric not in mdf.columns:
                continue

            x = mdf["severity"].astype(float)
            y = mdf[metric].astype(float)
            ax.plot(x, y, marker="o", markersize=4, linewidth=1.8, label=model_name)

            band_col = f"{metric}_std"
            if band_col in mdf.columns:
                band = mdf[band_col].astype(float)
                if band.notna().any():
                    ax.fill_between(x, y - band, y + band, alpha=0.18, label="_nolegend_")
            elif ci_low_col and ci_high_col and ci_low_col in mdf.columns and ci_high_col in mdf.columns:
                lo = mdf[ci_low_col].astype(float)
                hi = mdf[ci_high_col].astype(float)
                if lo.notna().any() and hi.notna().any():
                    ax.fill_between(x, lo, hi, alpha=0.18, label="_nolegend_")

            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        fam_human = corruption_family_name(str(family))
        ax.set_title(
            f"{title_main}\n{fam_human} · {subtitle_dataset}\nRun folder: {run_folder}",
            fontsize=10.5,
        )
        ax.set_xlabel(family_severity_xlabel(str(family)), fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.6)
        ax.legend(title="Model (config name)", fontsize=9, title_fontsize=9)
        fig.tight_layout()
        safe_family = str(family).replace("/", "_")
        fig.savefig(output / f"{file_stem}__{safe_family}.png", dpi=200)
        plt.close(fig)


def plot_degradation_curves(results_csv: str | Path, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_folder = output.name

    df = pd.read_csv(results_csv)
    if df.empty:
        return

    required = {"model", "corruption_family", "severity", "top1", "split"}
    if not required.issubset(set(df.columns)):
        return

    eval_df = df[df["split"] == "corrupted"].copy()
    if eval_df.empty:
        return

    subtitle = "ImageNet-1k validation subset (class-balanced)"

    plot_family_metric(
        eval_df,
        output,
        file_stem="plot_top1_accuracy_vs_corruption",
        metric="top1",
        ylabel="Top-1 accuracy (fraction of samples correct)",
        title_main="Top-1 accuracy vs corruption severity",
        subtitle_dataset=subtitle,
        run_folder=run_folder,
        ci_low_col="top1_ci_low",
        ci_high_col="top1_ci_high",
    )
    if "top5" in eval_df.columns:
        plot_family_metric(
            eval_df,
            output,
            file_stem="plot_top5_accuracy_vs_corruption",
            metric="top5",
            ylabel="Top-5 accuracy (fraction of samples correct)",
            title_main="Top-5 accuracy vs corruption severity",
            subtitle_dataset=subtitle,
            run_folder=run_folder,
            ci_low_col="top5_ci_low",
            ci_high_col="top5_ci_high",
        )
    if "nll_mean" in eval_df.columns:
        plot_family_metric(
            eval_df,
            output,
            file_stem="plot_mean_nll_vs_corruption",
            metric="nll_mean",
            ylabel="Mean negative log-likelihood (nats)",
            title_main="Classifier NLL vs corruption severity",
            subtitle_dataset=subtitle,
            run_folder=run_folder,
        )
    if "ece" in eval_df.columns:
        plot_family_metric(
            eval_df,
            output,
            file_stem="plot_ece_vs_corruption",
            metric="ece",
            ylabel="Expected calibration error (ECE)",
            title_main="Calibration (ECE) vs corruption severity",
            subtitle_dataset=subtitle,
            run_folder=run_folder,
        )

    if "robustness_ratio_top1" in df.columns:
        rdf = df[df["split"] == "corrupted"].copy()
        if not rdf.empty:
            plot_family_metric(
                rdf,
                output,
                file_stem="plot_top1_robustness_ratio_vs_corruption",
                metric="robustness_ratio_top1",
                ylabel="Robustness ratio (corrupted top-1 / clean top-1)",
                title_main="Relative robustness of top-1 accuracy",
                subtitle_dataset=subtitle,
                run_folder=run_folder,
            )


def plot_stability_curves(stability_csv: str | Path, output_dir: str | Path) -> None:
    output = Path(output_dir)
    run_folder = output.name
    p = Path(stability_csv)
    if not p.is_file():
        return
    df = pd.read_csv(p)
    if df.empty or "corruption_family" not in df.columns:
        return

    subtitle = "ImageNet-1k validation subset (class-balanced)"

    for family in sorted(df["corruption_family"].unique()):
        fam_df = df[df["corruption_family"] == family]
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for model_name in sorted(fam_df["model"].unique()):
            mdf = fam_df[fam_df["model"] == model_name].sort_values("severity")
            if mdf.empty or "stability_top1" not in mdf.columns:
                continue
            x = mdf["severity"].astype(float)
            y = mdf["stability_top1"].astype(float)
            ax.plot(x, y, marker="o", markersize=4, linewidth=1.8, label=model_name)
            band_col = "stability_top1_std"
            if band_col in mdf.columns:
                band = mdf[band_col].astype(float)
                if band.notna().any():
                    ax.fill_between(x, y - band, y + band, alpha=0.18, label="_nolegend_")

        fam_human = corruption_family_name(str(family))
        ax.set_title(
            f"Prediction stability vs corruption severity\n{fam_human} · {subtitle}\nRun folder: {run_folder}",
            fontsize=10.5,
        )
        ax.set_xlabel(family_severity_xlabel(str(family)), fontsize=10)
        ax.set_ylabel("Fraction of samples whose top-1 prediction matches clean", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.6)
        ax.legend(title="Model (config name)", fontsize=9, title_fontsize=9)
        fig.tight_layout()
        safe_family = str(family).replace("/", "_")
        fig.savefig(output / f"plot_prediction_stability_top1_vs_corruption__{safe_family}.png", dpi=200)
        plt.close(fig)


def plot_all_run_artifacts(run_dir: str | Path) -> None:
    run_path = Path(run_dir)
    res = run_path / out.EVAL_METRICS_MEAN_AND_STD_OVER_REPEATS_CSV
    if res.is_file():
        plot_degradation_curves(res, run_path)
    stab = run_path / out.PREDICTION_STABILITY_MEAN_AND_STD_OVER_REPEATS_CSV
    if stab.is_file():
        plot_stability_curves(stab, run_path)
