from __future__ import annotations

import numpy as np


def brightest_pixel(frame: np.ndarray) -> tuple[int, int, float]:
    arr = np.asarray(frame, dtype=float)
    y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return int(x), int(y), float(arr[y, x])
