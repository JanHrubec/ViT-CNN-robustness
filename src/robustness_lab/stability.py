from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def compute_prediction_stability(per_sample_rows: Iterable[dict]) -> list[dict]:
    """
    Prediction stability relative to clean

    Fraction of samples whose top-1 prediction matches clean prediction for the same model and dataset index
    """
    clean_pred: dict[tuple[str, int], int] = {}

    for row in per_sample_rows:
        if row.get("split") != "clean":
            continue
        model = row.get("model")
        idx = row.get("sample_index")
        pred = row.get("top1_pred")
        if model is None or idx is None or pred is None:
            continue
        clean_pred[(model, int(idx))] = int(pred)

    grouped: dict[tuple[str, str, float], list[int]] = defaultdict(list)

    for row in per_sample_rows:
        if row.get("split") != "corrupted":
            continue
        model = row.get("model")
        idx = row.get("sample_index")
        pred = row.get("top1_pred")
        family = row.get("corruption_family")
        severity = row.get("severity")
        if model is None or idx is None or pred is None or family is None or severity is None:
            continue
        key = (str(model), str(family), float(severity))
        clean = clean_pred.get((model, int(idx)))
        if clean is None:
            continue
        grouped[key].append(1 if int(pred) == clean else 0)

    out: list[dict] = []
    for (model, family, severity), hits in sorted(grouped.items()):
        if not hits:
            continue
        out.append(
            {
                "model": model,
                "corruption_family": family,
                "severity": severity,
                "stability_top1": sum(hits) / len(hits),
                "sample_count": len(hits),
            }
        )

    return out
