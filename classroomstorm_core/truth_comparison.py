from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Sequence


def read_truth_csv(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def nearest_truth_errors(localizations: Sequence[dict], truth_rows: Sequence[dict]) -> list[float]:
    truth_by_frame: dict[int, list[tuple[float, float]]] = {}
    unframed_truth: list[tuple[float, float]] = []
    for row in truth_rows:
        if str(row.get("active", "1")).lower() in ("0", "false"):
            continue
        point = (float(row["x_px"]), float(row["y_px"]))
        frame_value = str(row.get("frame", "")).strip()
        if frame_value:
            truth_by_frame.setdefault(int(float(frame_value)), []).append(point)
        else:
            unframed_truth.append(point)

    errors: list[float] = []
    for loc in localizations:
        x = float(loc.get("x_full_px", loc["x_px"]))
        y = float(loc.get("y_full_px", loc["y_px"]))
        frame_value = str(loc.get("frame", "")).strip()
        candidates = truth_by_frame.get(int(float(frame_value)), []) if frame_value else []
        if not candidates:
            candidates = unframed_truth
        if candidates:
            errors.append(min(math.hypot(x - tx, y - ty) for tx, ty in candidates))
    return errors


def match_localizations_to_truth(
    localizations: Sequence[dict], truth_rows: Sequence[dict],
    max_dist_px: float = 2.0, *, frames_processed: int | None = None,
    crop: dict | None = None,
) -> dict:
    """Frame-aware, one-to-one, maximum-cardinality matching within a radius.

    Candidates are visited in distance order. This maximizes the number of
    valid matches, not the global minimum distance of the resulting assignment.
    Unframed truth is interpreted as a static pattern in each evaluated frame.
    Crop and frame limits exclude truth outside the evaluated acquisition.
    """
    if not math.isfinite(max_dist_px) or max_dist_px <= 0:
        raise ValueError("truth matching radius must be finite and positive")
    locs_by_frame: dict[int, list[dict]] = {}
    for loc in localizations:
        frame = int(float(loc.get("frame", 0)))
        if frames_processed is None or 0 <= frame < frames_processed:
            locs_by_frame.setdefault(frame, []).append(loc)
    truth_by_frame: dict[int, list[dict]] = {}
    static = []
    for row in truth_rows:
        if str(row.get("active", "1")).lower() in ("0", "false"):
            continue
        x, y = float(row["x_px"]), float(row["y_px"])
        if crop and not (crop["x"] <= x < crop["x"] + crop["width"] and
                         crop["y"] <= y < crop["y"] + crop["height"]):
            continue
        value = str(row.get("frame", "")).strip()
        if not value:
            static.append(row)
        else:
            frame = int(float(value))
            if frames_processed is None or 0 <= frame < frames_processed:
                truth_by_frame.setdefault(frame, []).append(row)
    frames = set(range(frames_processed)) if frames_processed is not None else set(locs_by_frame) | set(truth_by_frame)
    if not frames and static:
        frames = {0}
    errors = []
    n_locs = n_truth = 0
    for frame in sorted(frames):
        locs = locs_by_frame.get(frame, [])
        truth = truth_by_frame.get(frame, []) + static
        n_locs += len(locs)
        n_truth += len(truth)
        candidates = []
        for loc in locs:
            x, y = float(loc.get("x_full_px", loc["x_px"])), float(loc.get("y_full_px", loc["y_px"]))
            pairs = [(math.hypot(x - float(t["x_px"]), y - float(t["y_px"])), ti)
                     for ti, t in enumerate(truth)]
            candidates.append(sorted((d, ti) for d, ti in pairs if d <= max_dist_px))
        assigned: dict[int, tuple[int, float]] = {}

        def augment(li: int, visited: set[int]) -> bool:
            for distance, ti in candidates[li]:
                if ti in visited:
                    continue
                visited.add(ti)
                if ti not in assigned or augment(assigned[ti][0], visited):
                    assigned[ti] = li, distance
                    return True
            return False

        for li in range(len(locs)):
            augment(li, set())
        errors.extend(distance for _, distance in assigned.values())
    tp = len(errors)
    return {
        "method": "frame_aware_maximum_cardinality_distance_ordered",
        "max_distance_px": float(max_dist_px),
        "coordinate_system": "original_frame",
        "matched_localizations": tp,
        "false_positives": n_locs - tp,
        "false_negatives": n_truth - tp,
        "evaluated_localizations": n_locs,
        "evaluated_truth_events": n_truth,
        "precision": tp / n_locs if n_locs else None,
        "recall": tp / n_truth if n_truth else None,
        "f1": 2 * tp / (n_locs + n_truth) if n_locs + n_truth else None,
        "mean_matched_error_px": statistics.mean(errors) if errors else None,
        "median_matched_error_px": statistics.median(errors) if errors else None,
        "max_matched_error_px": max(errors) if errors else None,
    }
