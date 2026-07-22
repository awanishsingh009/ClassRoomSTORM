from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence


def read_truth_csv(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def nearest_truth_errors(localizations: Sequence[dict], truth_rows: Sequence[dict]) -> list[float]:
    truth_by_frame: dict[int, list[tuple[float, float]]] = {}
    unframed_truth: list[tuple[float, float]] = []
    for row in truth_rows:
        if row.get("active", "1") in ("0", "False", "false"):
            continue
        point = (float(row["x_px"]), float(row["y_px"]))
        frame_value = str(row.get("frame", "")).strip()
        if frame_value:
            truth_by_frame.setdefault(int(float(frame_value)), []).append(point)
        else:
            unframed_truth.append(point)

    errors: list[float] = []
    for loc in localizations:
        x = float(loc["x_px"])
        y = float(loc["y_px"])
        frame_value = str(loc.get("frame", "")).strip()
        candidates = truth_by_frame.get(int(float(frame_value)), []) if frame_value else []
        if not candidates:
            candidates = unframed_truth
        if candidates:
            errors.append(min(math.hypot(x - tx, y - ty) for tx, ty in candidates))
    return errors
