from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_degradation_curves(results_csv: str | Path, output_dir: str | Path) -> None:
    """Top-1 degradation plot per corruption family."""
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

    for family in sorted(eval_df["corruption_family"].unique()):
        family_df = eval_df[eval_df["corruption_family"] == family]

        plt.figure(figsize=(7, 4.5))
        for model_name in sorted(family_df["model"].unique()):
            # Sort by severity
            mdf = family_df[family_df["model"] == model_name].sort_values("severity")
            plt.plot(mdf["severity"], mdf["top1"], marker="o", label=model_name)

        plt.title(f"Robustness curve: {family}")
        plt.xlabel("Severity")
        plt.ylabel("Top-1 Accuracy")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"curve_{family}.png", dpi=180)
        plt.close()
