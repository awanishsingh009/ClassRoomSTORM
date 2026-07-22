from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np


def require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required for video I/O. Install opencv-python.") from exc
    return cv2


def read_video_info(path: str | Path) -> dict:
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    return info


def read_video_frames(path: str | Path, max_frames: int | None = None) -> List[np.ndarray]:
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(float))
        if max_frames is not None and len(frames) >= int(max_frames):
            break
    cap.release()
    return frames


def write_mp4(path: str | Path, frames_uint8: Iterable[np.ndarray], fps: float = 12.0) -> Path:
    cv2 = require_cv2()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = [np.asarray(frame, dtype=np.uint8) for frame in frames_uint8]
    if not frames:
        raise ValueError("at least one frame is required for MP4 export")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {target}")
    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise ValueError("all frames must have the same shape")
        writer.write(frame)
    writer.release()
    return target
