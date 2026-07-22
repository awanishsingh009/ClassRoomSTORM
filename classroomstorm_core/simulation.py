from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BlinkerPoint:
    x_px: float
    y_px: float
    emitter_id: int = 0
    brightness_scale: float = 1.0
    label: str = ""


def compute_edge_padding_px(sigma_px: float, minimum_px: int = 20, sigma_factor: float = 5.0) -> int:
    """Return adaptive blank camera padding for a rendered Gaussian spot."""
    sigma = max(0.0, float(sigma_px))
    return int(max(int(minimum_px), math.ceil(float(sigma_factor) * sigma)))


def mask_to_camera_points(
    mask: np.ndarray,
    spacing_px: float,
    padding_px: int,
    brightness_scale: float = 1.0,
) -> Tuple[List[BlinkerPoint], Tuple[int, int]]:
    """Map nonzero source-mask pixels into padded camera-frame coordinates."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask must be a 2D array")
    spacing = float(spacing_px)
    if spacing <= 0:
        raise ValueError("spacing_px must be positive")
    pad = int(max(0, padding_px))
    rows, cols = np.nonzero(arr > 0)
    points: List[BlinkerPoint] = []
    for emitter_id, (row, col) in enumerate(zip(rows, cols)):
        points.append(
            BlinkerPoint(
                x_px=pad + float(col) * spacing,
                y_px=pad + float(row) * spacing,
                emitter_id=emitter_id,
                brightness_scale=float(brightness_scale),
            )
        )
    height = int(math.ceil((arr.shape[0] - 1) * spacing + 2 * pad + 1))
    width = int(math.ceil((arr.shape[1] - 1) * spacing + 2 * pad + 1))
    return points, (height, width)


def render_gaussian_frame(
    active_points: Sequence[BlinkerPoint],
    shape_px: Tuple[int, int],
    sigma_px: float,
    photons: float,
    background: float,
    read_noise: float,
    seed: int | None = None,
) -> np.ndarray:
    """Render one noisy camera frame from active Gaussian blinkers."""
    height, width = int(shape_px[0]), int(shape_px[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape_px must contain positive height and width")
    sigma = float(sigma_px)
    if sigma <= 0:
        raise ValueError("sigma_px must be positive")

    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width), dtype=float)
    for point in active_points:
        gauss = np.exp(-0.5 * (((xx - point.x_px) / sigma) ** 2 + ((yy - point.y_px) / sigma) ** 2))
        total = float(gauss.sum())
        if total > 0:
            gauss /= total
            image += float(photons) * float(point.brightness_scale) * gauss

    lam = np.clip(image + float(background), 0.0, None)
    rng = np.random.default_rng(seed)
    frame = rng.poisson(lam).astype(float)
    if float(read_noise) > 0:
        frame += rng.normal(0.0, float(read_noise), size=frame.shape)
    return np.clip(frame, 0.0, None)


def choose_active_points(
    all_points: Sequence[BlinkerPoint],
    rng: np.random.Generator,
    mode: str,
    blinkers_per_frame: int,
    spacing_mode: str = "random",
    min_distance_px: float = 0.0,
) -> List[BlinkerPoint]:
    """Choose active emitters for one frame."""
    if not all_points:
        return []
    count = 1 if mode == "single" else max(1, int(blinkers_per_frame))
    count = min(count, len(all_points))
    spacing_key = str(spacing_mode or "random").lower().replace(" ", "_")
    minimum = max(0.0, float(min_distance_px))
    if spacing_key in {"random", "none", "no_control"} or minimum <= 0 or count <= 1:
        idx = rng.choice(len(all_points), size=count, replace=False)
        return [all_points[int(i)] for i in np.atleast_1d(idx)]

    selected: List[BlinkerPoint] = []
    for idx in rng.permutation(len(all_points)):
        candidate = all_points[int(idx)]
        far_enough = True
        for chosen in selected:
            distance = math.hypot(candidate.x_px - chosen.x_px, candidate.y_px - chosen.y_px)
            if distance < minimum:
                far_enough = False
                break
        if far_enough:
            selected.append(candidate)
            if len(selected) >= count:
                break
    return selected


def write_truth_csv(path: str | Path, rows: Iterable[dict]) -> None:
    columns = ["frame", "emitter_id", "x_px", "y_px", "sigma_px", "brightness", "pattern_mode", "active", "note"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def normalize_to_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=float)
    if arr.size == 0:
        return arr.astype(np.uint8)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
