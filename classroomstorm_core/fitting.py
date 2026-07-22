from __future__ import annotations

import numpy as np


def centroid(frame: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(frame, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("cannot compute centroid of an empty frame")
    yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    return float((arr * xx).sum() / total), float((arr * yy).sum() / total)
