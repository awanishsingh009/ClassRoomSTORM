from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class ReconstructionResult:
    localizations: List[dict]
    raw_mean: np.ndarray
    superres: np.ndarray
    summary: dict


def _centroid_localization(frame: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    weights = np.asarray(frame, dtype=float) * mask
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("cannot localize an empty mask")
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    x = float((weights * xx).sum() / total)
    y = float((weights * yy).sum() / total)
    return x, y, total


def _localize_single(frame: np.ndarray, threshold_quantile: float, frame_index: int) -> dict | None:
    arr = np.asarray(frame, dtype=float)
    if arr.size == 0 or float(arr.max()) <= max(0.0, float(arr.min())):
        return None
    threshold = float(np.quantile(arr, float(threshold_quantile)))
    mask = arr >= threshold
    if not np.any(mask):
        y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
        mask = np.zeros(arr.shape, dtype=bool)
        mask[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = True
    x, y, intensity = _centroid_localization(arr, mask)
    return {
        "frame": int(frame_index),
        "emitter_id": 0,
        "x_px": x,
        "y_px": y,
        "intensity": intensity,
        "accepted": True,
    }


def _connected_components(binary: np.ndarray) -> list[np.ndarray]:
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    mask = np.asarray(binary, dtype=np.uint8)
    components: list[np.ndarray] = []
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) > 0:
                components.append(labels == label)
        return components

    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0 or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            comp = np.zeros(mask.shape, dtype=bool)
            while stack:
                cy, cx = stack.pop()
                comp[cy, cx] = True
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if not seen[ny, nx] and mask[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            components.append(comp)
    return components


def _component_localizations(frame: np.ndarray, threshold_quantile: float, frame_index: int, max_emitters: int = 20) -> list[dict]:
    arr = np.asarray(frame, dtype=float)
    threshold = float(np.quantile(arr, float(threshold_quantile)))
    return _component_localizations_from_mask(arr, arr >= threshold, frame_index, max_emitters=max_emitters)


def _component_localizations_from_mask(frame: np.ndarray, binary_mask: np.ndarray, frame_index: int, max_emitters: int = 20) -> list[dict]:
    arr = np.asarray(frame, dtype=float)
    if float(arr.max()) <= max(0.0, float(arr.min())):
        return []
    mask = np.asarray(binary_mask, dtype=bool)
    components = _connected_components(mask)
    ranked = []
    for comp in components:
        intensity = float((arr * comp).sum())
        area = int(comp.sum())
        if intensity > 0 and area >= 1:
            ranked.append((intensity, comp))
    ranked.sort(key=lambda item: item[0], reverse=True)
    locs: list[dict] = []
    for emitter_id, (_, comp) in enumerate(ranked[:max_emitters]):
        x, y, intensity = _centroid_localization(arr, comp)
        locs.append(
            {
                "frame": int(frame_index),
                "emitter_id": int(emitter_id),
                "x_px": x,
                "y_px": y,
                "intensity": intensity,
                "accepted": True,
            }
        )
    return locs


def _estimate_blinker_mode(frames: Sequence[np.ndarray], threshold_quantile: float, scan_frames: int = 40) -> str:
    counts = []
    for frame in frames[: max(1, int(scan_frames))]:
        counts.append(len(_component_localizations(frame, threshold_quantile, 0, max_emitters=50)))
    if counts and float(np.median(counts)) > 1.0:
        return "multiple"
    return "single"


def _localize_frame_task(args: tuple[int, np.ndarray, float, str]) -> list[dict]:
    idx, frame, threshold_quantile, resolved_mode = args
    if resolved_mode == "multiple":
        return _component_localizations(frame, threshold_quantile, idx)
    loc = _localize_single(frame, threshold_quantile, idx)
    return [loc] if loc is not None else []


def _localize_frame_with_mask_task(args: tuple[int, np.ndarray, np.ndarray]) -> list[dict]:
    idx, frame, mask = args
    return _component_localizations_from_mask(frame, mask, idx)


def _default_worker_count() -> int:
    return max(1, int(float(os.cpu_count() or 1) * 0.9))


def _torch_cuda_status() -> tuple[bool, str]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return False, f"PyTorch is not installed: {exc}"
    try:
        if not torch.cuda.is_available():
            return False, "PyTorch is installed, but CUDA is not available."
        return True, str(torch.cuda.get_device_name(0))
    except Exception as exc:
        return False, f"PyTorch CUDA check failed: {exc}"


def _resolve_processing_backend(
    requested_backend: str,
    frame_count: int,
    worker_count: int | None,
    resolved_mode: str,
) -> tuple[str, int, str | None, str | None]:
    requested = str(requested_backend or "serial").lower().replace(" ", "_").replace("-", "_")
    workers = int(worker_count) if worker_count is not None and int(worker_count) > 0 else _default_worker_count()
    workers = max(1, min(workers, int(frame_count)))
    if requested in {"auto_cpu_gpu_acceleration", "auto_cpu_gpu_accelerated", "cpu_gpu_acceleration"}:
        fallback = "cpu_parallel" if workers > 1 and int(frame_count) > 1 else "serial"
        available, detail = _torch_cuda_status()
        if available and resolved_mode == "single":
            return "gpu_torch", 1, None, detail
        if available and resolved_mode == "multiple":
            return "gpu_threshold_cpu_components", workers, "Auto CPU+GPU acceleration used GPU batch thresholding and CPU connected-component localization.", detail
        return fallback, workers, f"Auto CPU+GPU acceleration requested CUDA, but it was unavailable: {detail} Reconstruction fell back to CPU processing.", None
    if requested in {"auto_cpu_gpu", "auto_gpu", "auto_accelerated"}:
        fallback = "cpu_parallel" if workers > 1 and int(frame_count) > 1 else "serial"
        if resolved_mode != "single":
            return fallback, workers, "Auto CPU+GPU selected CPU parallel for multiple-emitter connected-component localization; GPU is currently used automatically for single-emitter localization when available.", None
        available, detail = _torch_cuda_status()
        if available:
            return "gpu_torch", 1, None, detail
        return fallback, workers, f"Auto CPU+GPU requested GPU for single-emitter localization, but CUDA was unavailable: {detail} Reconstruction fell back to CPU processing.", None
    if requested in {"auto", "auto_cpu"}:
        used = "cpu_parallel" if workers > 1 and int(frame_count) > 1 else "serial"
        return used, workers, None, None
    if requested in {"cpu", "cpu_parallel", "parallel"}:
        used = "cpu_parallel" if workers > 1 and int(frame_count) > 1 else "serial"
        return used, workers, None, None
    if requested in {"hybrid", "hybrid_cpu_gpu", "cpu_gpu"}:
        fallback = "cpu_parallel" if workers > 1 and int(frame_count) > 1 else "serial"
        if resolved_mode != "single":
            return fallback, workers, "Hybrid CPU+GPU currently uses CPU parallel for multiple-emitter connected-component localization; GPU is used for single-emitter localization when available.", None
        available, detail = _torch_cuda_status()
        if available:
            return "gpu_torch", 1, None, detail
        return fallback, workers, f"Hybrid CPU+GPU requested GPU for single-emitter localization, but CUDA was unavailable: {detail} Reconstruction fell back to CPU processing.", None
    if requested in {"gpu", "gpu_only", "only_gpu", "gpu_experimental"}:
        fallback = "cpu_parallel" if workers > 1 and int(frame_count) > 1 else "serial"
        if resolved_mode != "single":
            return fallback, workers, "GPU experimental currently supports single-emitter localization only; multiple-emitter reconstruction fell back to CPU processing.", None
        available, detail = _torch_cuda_status()
        if available:
            return "gpu_torch", 1, None, detail
        return fallback, workers, f"GPU experimental requires PyTorch with CUDA; {detail} Reconstruction fell back to CPU processing.", None
    if requested in {"serial", "single_core"}:
        return "serial", 1, None, None
    raise ValueError("processing_backend must be serial, auto_cpu, gpu_only, auto_cpu_gpu, auto_cpu_gpu_acceleration, cpu_parallel, hybrid_cpu_gpu, or gpu_experimental")


def _localize_frames(
    frame_list: Sequence[np.ndarray],
    threshold_quantile: float,
    resolved_mode: str,
    processing_backend: str,
    worker_count: int,
) -> list[dict]:
    tasks = [(idx, frame, threshold_quantile, resolved_mode) for idx, frame in enumerate(frame_list)]
    if processing_backend != "cpu_parallel":
        localizations: list[dict] = []
        for task in tasks:
            localizations.extend(_localize_frame_task(task))
        return localizations

    localizations = []
    with ProcessPoolExecutor(max_workers=max(1, int(worker_count))) as executor:
        for locs in executor.map(_localize_frame_task, tasks):
            localizations.extend(locs)
    return localizations


def _localize_frames_gpu_single(
    frame_list: Sequence[np.ndarray],
    threshold_quantile: float,
    batch_size: int = 128,
) -> list[dict]:
    import torch  # type: ignore

    localizations: list[dict] = []
    if not frame_list:
        return localizations
    height, width = frame_list[0].shape
    device = torch.device("cuda")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    q = float(threshold_quantile)
    for start in range(0, len(frame_list), max(1, int(batch_size))):
        chunk = frame_list[start:start + max(1, int(batch_size))]
        stack = np.stack(chunk, axis=0).astype(np.float32, copy=False)
        tensor = torch.as_tensor(stack, device=device)
        flat = tensor.reshape(tensor.shape[0], -1)
        thresholds = torch.quantile(flat, q, dim=1).reshape(-1, 1, 1)
        has_signal = (flat.max(dim=1).values > flat.min(dim=1).values).reshape(-1, 1, 1)
        mask = (tensor >= thresholds) & has_signal
        weights = tensor * mask
        totals = weights.sum(dim=(1, 2))
        x_vals = (weights * xx).sum(dim=(1, 2)) / torch.clamp(totals, min=1e-12)
        y_vals = (weights * yy).sum(dim=(1, 2)) / torch.clamp(totals, min=1e-12)
        totals_cpu = totals.detach().cpu().numpy()
        xs_cpu = x_vals.detach().cpu().numpy()
        ys_cpu = y_vals.detach().cpu().numpy()
        for offset, total in enumerate(totals_cpu):
            if float(total) <= 0:
                continue
            localizations.append(
                {
                    "frame": int(start + offset),
                    "emitter_id": 0,
                    "x_px": float(xs_cpu[offset]),
                    "y_px": float(ys_cpu[offset]),
                    "intensity": float(total),
                    "accepted": True,
                }
            )
    torch.cuda.synchronize()
    return localizations


def _localize_frames_gpu_threshold_cpu_components(
    frame_list: Sequence[np.ndarray],
    threshold_quantile: float,
    worker_count: int,
    batch_size: int = 128,
) -> list[dict]:
    import torch  # type: ignore

    masks: list[np.ndarray] = []
    q = float(threshold_quantile)
    device = torch.device("cuda")
    for start in range(0, len(frame_list), max(1, int(batch_size))):
        chunk = frame_list[start:start + max(1, int(batch_size))]
        stack = np.stack(chunk, axis=0).astype(np.float32, copy=False)
        tensor = torch.as_tensor(stack, device=device)
        flat = tensor.reshape(tensor.shape[0], -1)
        thresholds = torch.quantile(flat, q, dim=1).reshape(-1, 1, 1)
        batch_masks = (tensor >= thresholds).detach().cpu().numpy().astype(bool)
        masks.extend([batch_masks[i] for i in range(batch_masks.shape[0])])
    torch.cuda.synchronize()

    tasks = [(idx, frame, masks[idx]) for idx, frame in enumerate(frame_list)]
    localizations: list[dict] = []
    if int(worker_count) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=max(1, int(worker_count))) as executor:
            for locs in executor.map(_localize_frame_with_mask_task, tasks):
                localizations.extend(locs)
        return localizations
    for task in tasks:
        localizations.extend(_localize_frame_with_mask_task(task))
    return localizations


def normalize_crop(crop: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Intersect an integer crop rectangle with the original camera frame."""
    if len(crop) != 4 or any(not np.isfinite(v) or int(v) != v for v in crop):
        raise ValueError("crop must contain four finite integers: x, y, width, height")
    x, y, width, height = map(int, crop)
    h, w = shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + width), min(h, y + height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("crop region is empty")
    return x0, y0, x1 - x0, y1 - y0


def _apply_crop(frame: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return frame
    x0, y0, width, height = normalize_crop(crop, frame.shape)
    x1, y1 = x0 + width, y0 + height
    return frame[y0:y1, x0:x1]


def render_count_image(localizations: Sequence[dict], shape: tuple[int, int]) -> np.ndarray:
    image = np.zeros(shape, dtype=float)
    height, width = shape
    for loc in localizations:
        x = int(round(float(loc["x_px"])))
        y = int(round(float(loc["y_px"])))
        if 0 <= x < width and 0 <= y < height:
            image[y, x] += 1.0
    return image


def render_localization_image(localizations: Sequence[dict], shape: tuple[int, int], sigma_px: float = 1.5) -> np.ndarray:
    """Sample Gaussian spots at fractional coordinates on the camera grid.

    Each localization contributes one count, including at image boundaries.
    The display width is not an estimate of uncertainty or optical resolution.
    """
    image = np.zeros(shape, dtype=float)
    height, width = shape
    sigma = float(sigma_px)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("render sigma must be finite and positive")
    radius = max(1, int(np.ceil(4 * sigma)))
    for loc in localizations:
        x, y = float(loc["x_px"]), float(loc["y_px"])
        if not (0 <= x < width and 0 <= y < height):
            continue
        y0, y1 = max(0, int(y) - radius), min(height, int(y) + radius + 2)
        x0, x1 = max(0, int(x) - radius), min(width, int(x) + radius + 2)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        kernel = np.exp(-0.5 * (((xx - x) / sigma) ** 2 + ((yy - y) / sigma) ** 2))
        image[y0:y1, x0:x1] += kernel / kernel.sum()
    return image


def reconstruct_frames(
    frames: Iterable[np.ndarray],
    threshold_quantile: float = 0.995,
    multi_emitter: bool = False,
    blinker_mode: str | None = None,
    crop: tuple[int, int, int, int] | None = None,
    processing_backend: str = "serial",
    worker_count: int | None = None,
    pixel_size: float | None = None,
    unit: str = "px",
    background_mode: str = "none",
) -> ReconstructionResult:
    if not np.isfinite(threshold_quantile) or not 0 < threshold_quantile <= 1:
        raise ValueError("threshold_quantile must be finite and in (0, 1]")
    if pixel_size is not None and (not np.isfinite(pixel_size) or pixel_size <= 0):
        raise ValueError("pixel_size must be finite and positive")
    if not str(unit).strip():
        raise ValueError("calibration unit must not be empty")
    if background_mode not in {"none", "frame_median"}:
        raise ValueError("background_mode must be none or frame_median")
    frame_list = []
    input_shape = None
    effective_crop = None
    raw_sum = None
    backgrounds = []
    for index, frame in enumerate(frames):
        arr = np.asarray(frame)
        if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
            raise ValueError(f"frame {index} must be a nonempty, finite 2D grayscale image")
        if input_shape is None:
            input_shape = arr.shape
            effective_crop = normalize_crop(crop, input_shape) if crop is not None else None
        elif arr.shape != input_shape:
            raise ValueError("all frames must have the same shape")
        arr = _apply_crop(arr, effective_crop).astype(np.float32)
        if raw_sum is None:
            raw_sum = np.zeros(arr.shape, dtype=np.float64)
        raw_sum += arr
        background = float(np.median(arr)) if background_mode == "frame_median" else 0.0
        backgrounds.append(background)
        frame_list.append(np.maximum(arr - background, 0.0))
    if not frame_list:
        raise ValueError("at least one frame is required")
    shape = frame_list[0].shape
    if any(frame.shape != shape for frame in frame_list):
        raise ValueError("all frames must have the same shape")

    requested_mode = (blinker_mode or ("multiple" if multi_emitter else "single")).lower()
    if requested_mode not in {"single", "multiple", "auto"}:
        raise ValueError("blinker_mode must be single, multiple, or auto")
    resolved_mode = _estimate_blinker_mode(frame_list, threshold_quantile) if requested_mode == "auto" else requested_mode

    t0 = time.perf_counter()
    used_backend, used_workers, backend_note, gpu_device = _resolve_processing_backend(processing_backend, len(frame_list), worker_count, resolved_mode)
    t_localize0 = time.perf_counter()
    if used_backend == "gpu_torch":
        localizations = _localize_frames_gpu_single(frame_list, threshold_quantile)
    elif used_backend == "gpu_threshold_cpu_components":
        localizations = _localize_frames_gpu_threshold_cpu_components(frame_list, threshold_quantile, used_workers)
    else:
        localizations = _localize_frames(frame_list, threshold_quantile, resolved_mode, used_backend, used_workers)
    t_localize1 = time.perf_counter()

    x_offset, y_offset = effective_crop[:2] if effective_crop else (0, 0)
    for loc in localizations:
        loc["x_full_px"] = loc["x_px"] + x_offset
        loc["y_full_px"] = loc["y_px"] + y_offset
        if pixel_size is not None:
            loc.update(x_calibrated=loc["x_full_px"] * pixel_size,
                       y_calibrated=loc["y_full_px"] * pixel_size, unit=str(unit).strip())
    raw_mean = raw_sum / len(frame_list)
    superres = render_localization_image(localizations, shape, sigma_px=1.5)
    t_done = time.perf_counter()
    summary = {
        "frames_processed": len(frame_list),
        "localizations": len(localizations),
        "multi_emitter_requested": bool(multi_emitter),
        "blinker_mode_requested": requested_mode,
        "blinker_mode_resolved": resolved_mode,
        "threshold_quantile": float(threshold_quantile),
        "background_mode": background_mode,
        "mean_subtracted_background": float(np.mean(backgrounds)),
        "coordinate_system": "x_px/y_px are crop-local; x_full_px/y_full_px use the original frame origin",
        "calibration": {"applied": pixel_size is not None, "unit_per_pixel": pixel_size, "unit": str(unit).strip(), "origin": "original_frame"},
        "rendering": {"method": "fractional_coordinate_gaussian", "sigma_px": 1.5, "grid": "camera_pixels", "normalization": "one_count_per_localization", "measured_resolution": False},
        "frames_without_localizations": len(frame_list) - len({loc["frame"] for loc in localizations}),
        "processing_backend_requested": str(processing_backend or "serial").lower().replace(" ", "_").replace("-", "_"),
        "processing_backend_used": used_backend,
        "worker_count": int(used_workers if used_backend in {"cpu_parallel", "gpu_threshold_cpu_components"} else 1),
        "cpu_count": int(os.cpu_count() or 1),
        "timing_seconds": {
            "localization": float(t_localize1 - t_localize0),
            "rendering_and_summary": float(t_done - t_localize1),
            "total_reconstruct_frames": float(t_done - t0),
        },
    }
    if backend_note:
        summary["gpu_note"] = backend_note
    if gpu_device:
        summary["gpu_device"] = gpu_device
    if effective_crop is not None:
        x, y, width, height = effective_crop
        summary["crop"] = {"x": x, "y": y, "width": width, "height": height}
    return ReconstructionResult(localizations=localizations, raw_mean=raw_mean, superres=superres, summary=summary)


def write_localizations_csv(path: str | Path, localizations: Sequence[dict]) -> None:
    columns = ["frame", "emitter_id", "x_px", "y_px", "intensity", "accepted"]
    columns += ["x_full_px", "y_full_px", "x_calibrated", "y_calibrated", "unit"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for loc in localizations:
            writer.writerow({key: loc.get(key, "") for key in columns})


def write_summary_json(path: str | Path, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
