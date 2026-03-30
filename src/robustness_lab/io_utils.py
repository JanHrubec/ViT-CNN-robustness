from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    """UTC timestamp for stable, sortable run directory names."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_json(path: str | Path, payload: Any) -> None:
    """Save JSON payload, auto-creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload):
        payload = asdict(payload)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save list-of-dicts as CSV; writes empty file when no rows exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with p.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
