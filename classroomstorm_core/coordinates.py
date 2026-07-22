from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    width: int
    height: int


def cropped_to_full(x_px: float, y_px: float, crop: Crop | None) -> tuple[float, float]:
    if crop is None:
        return float(x_px), float(y_px)
    return float(x_px) + float(crop.x), float(y_px) + float(crop.y)
