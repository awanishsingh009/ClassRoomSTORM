from __future__ import annotations

import numpy as np


def clamp_point(shape: tuple[int, int], x_px: int | float, y_px: int | float) -> tuple[int, int]:
    """Clamp a requested image coordinate to a 2D array shape."""
    height, width = [int(v) for v in shape[:2]]
    if height <= 0 or width <= 0:
        raise ValueError("profile images must have non-empty height and width")
    x = max(0, min(width - 1, int(round(float(x_px)))))
    y = max(0, min(height - 1, int(round(float(y_px)))))
    return x, y


def normalize_profile(values: np.ndarray) -> np.ndarray:
    """Return a float profile scaled to 0..1, preserving flat profiles as zeros."""
    profile = np.asarray(values, dtype=float)
    if profile.size == 0:
        return profile
    low = float(np.min(profile))
    high = float(np.max(profile))
    if high <= low:
        return np.zeros_like(profile, dtype=float)
    return (profile - low) / (high - low)


def linked_line_profiles(raw_image: np.ndarray, reconstruction: np.ndarray, x_px: int | float, y_px: int | float) -> dict:
    """Extract normalized horizontal and vertical profiles at a linked pixel."""
    raw = np.asarray(raw_image, dtype=float)
    recon = np.asarray(reconstruction, dtype=float)
    if raw.ndim != 2 or recon.ndim != 2:
        raise ValueError("linked profiles require 2D raw and reconstruction images")

    x, y = clamp_point(raw.shape, x_px, y_px)
    recon_x, recon_y = clamp_point(recon.shape, x, y)

    return {
        "point": {"x_px": x, "y_px": y},
        "raw_x": normalize_profile(raw[y, :]),
        "recon_x": normalize_profile(recon[recon_y, :]),
        "raw_y": normalize_profile(raw[:, x]),
        "recon_y": normalize_profile(recon[:, recon_x]),
    }


def linked_profile_series() -> list[dict]:
    """Return display styles for linked raw/reconstruction profile curves."""
    return [
        {
            "key": "raw",
            "label": "Raw mean",
            "color": "#c62828",
        },
        {
            "key": "reconstruction",
            "label": "Reconstruction",
            "color": "#1565c0",
        },
    ]
