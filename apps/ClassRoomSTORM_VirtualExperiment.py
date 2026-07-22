"""
ClassRoomSTORM Virtual Experiment V1.

This script is the student-facing virtual experiment app. It keeps the main
workflow simple: choose a pattern, generate blinking frames, and export an MP4
plus truth/metadata files for later independent reconstruction.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED = PROJECT_ROOT / "classroomstorm_core"
sys.path.insert(0, str(SHARED))

from metadata import write_metadata
from led_text import CELL_SIZE, text_to_led_mask
from simulation import (
    compute_edge_padding_px,
    choose_active_points,
    mask_to_camera_points,
    normalize_to_uint8,
    render_gaussian_frame,
    write_truth_csv,
)
from video_io import read_video_frames, write_mp4


APP_NAME = "ClassRoomSTORM Virtual Experiment"
APP_VERSION = "V1.1"
AUTHOR_CREDIT = "Developed by Dr. Awanish Pratap Singh"
AUTHOR_AFFILIATION = "Institute of Biomedical Optics, University of Lübeck"
TEXT_MASK_MODE = "fixed_16x16_led_alphabet"
DEFAULT_TEXT_SIGMA_PX = 10.0
DEFAULT_TEXT_FRAMES = 1000
DEFAULT_TEXT_PHOTONS = 80000.0
DEFAULT_TEXT_BACKGROUND = 1.0
DEFAULT_TEXT_READ_NOISE = 0.2
DEFAULT_TEXT_SPACING_PX = 2.6
DEFAULT_BLINKERS_PER_FRAME = 30
DEFAULT_MANUAL_MIN_DISTANCE_PX = 100.0
DEFAULT_LOGO_SCALE = 1.0
DEFAULT_LOGO_SAMPLING = "grid"
DEFAULT_LOGO_GRID_STEP_PX = 1
VIRTUAL_FRAME_BACKENDS = ("auto_cpu", "cpu_parallel", "gpu_cuda")


def current_timestamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def safe_name_part(text: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or fallback


def virtual_default_output_dir(pattern: str, text: str = "", timestamp: str | None = None) -> Path:
    stamp = timestamp or current_timestamp()
    mode = str(pattern or "").lower()
    if mode == "logo":
        label = "logo"
    elif mode == "dots":
        label = "dot"
    else:
        label = safe_name_part(text, "text")
    return PROJECT_ROOT / "results" / "virtual_experiments" / f"ClassRoomSTORM_VirtualExperiment_{label}_{stamp}"


def studio_launcher_script() -> Path:
    return PROJECT_ROOT / "apps" / "ClassRoomSTORM_Studio.py"


def default_virtual_settings() -> dict:
    return {
        "sigma_px": DEFAULT_TEXT_SIGMA_PX,
        "frames": DEFAULT_TEXT_FRAMES,
        "photons": DEFAULT_TEXT_PHOTONS,
        "background": DEFAULT_TEXT_BACKGROUND,
        "read_noise": DEFAULT_TEXT_READ_NOISE,
        "spacing_px": DEFAULT_TEXT_SPACING_PX,
        "blinkers_per_frame": DEFAULT_BLINKERS_PER_FRAME,
        "manual_min_distance_px": DEFAULT_MANUAL_MIN_DISTANCE_PX,
        "logo_scale": DEFAULT_LOGO_SCALE,
        "logo_sampling": DEFAULT_LOGO_SAMPLING,
        "logo_grid_step_px": DEFAULT_LOGO_GRID_STEP_PX,
    }


def blinker_min_distance_px(spacing_mode: str, sigma_px: float, manual_distance_px: float = 0.0) -> float:
    mode = str(spacing_mode or "separated").lower().replace(" ", "_")
    manual = max(0.0, float(manual_distance_px))
    sigma = max(0.0, float(sigma_px))
    if mode in {"random", "none", "no_control"}:
        return 0.0
    if manual > 0:
        if mode == "overlapping":
            return 0.75 * manual
        if mode == "touching":
            return manual
        return 1.5 * manual
    if mode == "overlapping":
        return 1.0 * sigma
    if mode == "touching":
        return 2.0 * sigma
    return 3.0 * sigma


def blinker_spacing_multiplier(spacing_mode: str) -> float:
    mode = str(spacing_mode or "separated").lower().replace(" ", "_")
    if mode in {"random", "none", "no_control"}:
        return 0.0
    if mode == "overlapping":
        return 0.75
    if mode == "touching":
        return 1.0
    return 1.5


def effective_photons_per_blinker(base_photons: float, blinkers_per_frame: int) -> float:
    scale = max(1.0, float(max(1, int(blinkers_per_frame))) / 4.0)
    return float(base_photons) * scale


def virtual_worker_count(requested: int, frame_count: int) -> int:
    """Return the number of CPU workers to use for virtual frame rendering."""
    frames = max(1, int(frame_count))
    if int(requested) > 0:
        return max(1, min(int(requested), frames))
    auto_workers = max(1, int(float(os.cpu_count() or 1) * 0.9))
    return max(1, min(auto_workers, frames))


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


def resolve_virtual_frame_backend(
    requested_backend: str,
    worker_count: int,
    frame_count: int,
) -> tuple[str, int, str | None, str | None]:
    requested = str(requested_backend or "auto_cpu").lower().replace(" ", "_").replace("-", "_")
    workers = virtual_worker_count(worker_count, frame_count)
    cpu_used = "cpu_parallel" if workers > 1 and int(frame_count) > 1 else "serial"
    if requested in {"auto", "auto_cpu"}:
        return cpu_used, workers, None, None
    if requested in {"cpu", "cpu_parallel", "parallel"}:
        return cpu_used, workers, None, None
    if requested in {"gpu", "gpu_cuda", "cuda"}:
        available, detail = _torch_cuda_status()
        if available:
            return "gpu_cuda", 1, None, detail
        return cpu_used, workers, f"GPU CUDA requested, but unavailable: {detail} Virtual generation fell back to CPU.", None
    raise ValueError("frame_backend must be auto_cpu, cpu_parallel, or gpu_cuda")


def _render_virtual_frame_job(job: dict) -> dict:
    frame = render_gaussian_frame(
        job["active"],
        job["canvas_shape"],
        sigma_px=job["sigma_px"],
        photons=job["frame_photons"],
        background=job["background"],
        read_noise=job["read_noise"],
        seed=job["render_seed"],
    )
    truth_rows = []
    for point in job["active"]:
        truth_rows.append(
            {
                "frame": job["frame_index"],
                "emitter_id": point.emitter_id,
                "x_px": point.x_px,
                "y_px": point.y_px,
                "sigma_px": job["sigma_px"],
                "brightness": job["frame_photons"] * point.brightness_scale,
                "pattern_mode": job["pattern"],
                "active": 1,
                "note": "padded_camera_coordinates",
            }
        )
    return {
        "frame_index": job["frame_index"],
        "active_count": len(job["active"]),
        "preview_frame": frame if int(job["frame_index"]) == 0 else None,
        "video_frame": normalize_to_uint8(frame),
        "truth_rows": truth_rows,
    }


def _render_gaussian_frame_gpu_cuda(
    active_points,
    shape_px: tuple[int, int],
    sigma_px: float,
    photons: float,
    background: float,
    read_noise: float,
    seed: int | None = None,
) -> np.ndarray:
    import torch  # type: ignore

    height, width = int(shape_px[0]), int(shape_px[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape_px must contain positive height and width")
    sigma = float(sigma_px)
    if sigma <= 0:
        raise ValueError("sigma_px must be positive")

    device = torch.device("cuda")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    image = torch.zeros((height, width), device=device, dtype=torch.float32)
    for point in active_points:
        gauss = torch.exp(-0.5 * (((xx - float(point.x_px)) / sigma) ** 2 + ((yy - float(point.y_px)) / sigma) ** 2))
        total = torch.sum(gauss)
        if float(total.detach().cpu()) > 0:
            image += float(photons) * float(point.brightness_scale) * gauss / total

    lam = torch.clamp(image + float(background), min=0.0)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed))
    frame = torch.poisson(lam, generator=generator).to(torch.float32)
    if float(read_noise) > 0:
        noise = torch.normal(0.0, float(read_noise), size=frame.shape, device=device, generator=generator)
        frame = frame + noise
    frame = torch.clamp(frame, min=0.0)
    torch.cuda.synchronize()
    return frame.detach().cpu().numpy()


def _render_virtual_frame_job_gpu_cuda(job: dict) -> dict:
    frame = _render_gaussian_frame_gpu_cuda(
        job["active"],
        job["canvas_shape"],
        sigma_px=job["sigma_px"],
        photons=job["frame_photons"],
        background=job["background"],
        read_noise=job["read_noise"],
        seed=job["render_seed"],
    )
    truth_rows = []
    for point in job["active"]:
        truth_rows.append(
            {
                "frame": job["frame_index"],
                "emitter_id": point.emitter_id,
                "x_px": point.x_px,
                "y_px": point.y_px,
                "sigma_px": job["sigma_px"],
                "brightness": job["frame_photons"] * point.brightness_scale,
                "pattern_mode": job["pattern"],
                "active": 1,
                "note": "padded_camera_coordinates",
            }
        )
    return {
        "frame_index": job["frame_index"],
        "active_count": len(job["active"]),
        "preview_frame": frame if int(job["frame_index"]) == 0 else None,
        "video_frame": normalize_to_uint8(frame),
        "truth_rows": truth_rows,
    }


def virtual_result_catalog(output_dir: str | Path, include_existing: bool = True) -> list[dict]:
    """Return existing virtual experiment outputs grouped for the student browser."""
    out = Path(output_dir)
    sections = [
        {
            "key": "source",
            "label": "Source / Mask",
            "items": [
                ("Pattern Mask", "pattern_mask.png", "image"),
            ],
        },
        {
            "key": "sample",
            "label": "Sample Frames",
            "items": [
                ("Sample Blink Frame", "preview_frame.png", "image"),
            ],
        },
        {
            "key": "files",
            "label": "Generated Files",
            "items": [
                ("Blinking Video", "blinking_video.mp4", "video"),
                ("Truth CSV", "truth.csv", "text"),
                ("Metadata JSON", "metadata.json", "text"),
            ],
        },
    ]
    catalog = []
    for section in sections:
        items = []
        if include_existing:
            for label, filename, kind in section["items"]:
                path = out / filename
                if path.exists():
                    items.append({"label": label, "path": path, "kind": kind})
        catalog.append({"key": section["key"], "label": section["label"], "items": items})
    return catalog


def make_dot_mask(mode: str, size: int = 16) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    if mode == "single":
        mask[size // 2, size // 2] = 1
    elif mode == "pair":
        mask[size // 2, size // 2 - 2] = 1
        mask[size // 2, size // 2 + 2] = 1
    else:
        rng = np.random.default_rng(123)
        idx = rng.choice(size * size, size=max(6, size), replace=False)
        mask.reshape(-1)[idx] = 1
    return mask


def make_text_mask(text: str, width: int = 160, height: int = 64) -> np.ndarray:
    return text_to_led_mask(text)


def scale_binary_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.uint8)
    factor = max(0.05, float(scale))
    height = max(1, int(round(arr.shape[0] * factor)))
    width = max(1, int(round(arr.shape[1] * factor)))
    image = Image.fromarray((arr > 0).astype(np.uint8) * 255)
    resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)


def sample_logo_source_mask(
    mask: np.ndarray,
    method: str = "grid",
    grid_step_px: int = 4,
    max_source_points: int = 0,
    seed: int = 2026,
) -> np.ndarray:
    arr = (np.asarray(mask) > 0).astype(np.uint8)
    method_key = str(method or "grid").lower().replace(" ", "_")
    if method_key in {"all", "all_pixels"}:
        return arr

    points_y, points_x = np.where(arr > 0)
    if len(points_x) == 0:
        return arr

    sampled = np.zeros(arr.shape, dtype=np.uint8)
    if method_key in {"random", "random_max_points"}:
        limit = int(max_source_points) if int(max_source_points) > 0 else min(len(points_x), 5000)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(points_x), size=min(limit, len(points_x)), replace=False)
        sampled[points_y[chosen], points_x[chosen]] = 1
        return sampled

    if method_key not in {"grid", "grid_spacing"}:
        raise ValueError("source sampling must be all_pixels, grid, or random_max_points")

    step = max(1, int(grid_step_px))
    for y0 in range(0, arr.shape[0], step):
        for x0 in range(0, arr.shape[1], step):
            tile = arr[y0:y0 + step, x0:x0 + step]
            local_y, local_x = np.where(tile > 0)
            if len(local_x) == 0:
                continue
            center_y = (tile.shape[0] - 1) / 2
            center_x = (tile.shape[1] - 1) / 2
            best = int(np.argmin((local_y - center_y) ** 2 + (local_x - center_x) ** 2))
            sampled[y0 + int(local_y[best]), x0 + int(local_x[best])] = 1

    if int(max_source_points) > 0 and int(sampled.sum()) > int(max_source_points):
        sy, sx = np.where(sampled > 0)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(sx), size=int(max_source_points), replace=False)
        reduced = np.zeros(arr.shape, dtype=np.uint8)
        reduced[sy[chosen], sx[chosen]] = 1
        return reduced
    return sampled


def make_logo_mask(path: str | Path, target_size: int = 256, invert: bool = False) -> np.ndarray:
    image = Image.open(path).convert("L")
    image.thumbnail((target_size, target_size))
    canvas = Image.new("L", (target_size, target_size), 255)
    x = (target_size - image.width) // 2
    y = (target_size - image.height) // 2
    canvas.paste(image, (x, y))
    arr = np.asarray(canvas)
    threshold = float(arr.mean())
    mask = arr < threshold
    if invert:
        mask = ~mask
    return mask.astype(np.uint8)


def build_source_mask(
    pattern: str,
    text: str,
    logo_path: str,
    dot_mode: str,
    logo_scale: float = DEFAULT_LOGO_SCALE,
    logo_sampling: str = DEFAULT_LOGO_SAMPLING,
    logo_grid_step_px: int = DEFAULT_LOGO_GRID_STEP_PX,
    logo_max_source_points: int = 0,
    seed: int = 2026,
) -> np.ndarray:
    pattern = pattern.lower()
    if pattern == "dots":
        return make_dot_mask(dot_mode)
    if pattern == "text":
        return make_text_mask(text)
    if pattern == "logo":
        if not logo_path:
            raise ValueError("logo mode requires --logo")
        mask = scale_binary_mask(make_logo_mask(logo_path), logo_scale)
        return sample_logo_source_mask(mask, method=logo_sampling, grid_step_px=logo_grid_step_px, max_source_points=logo_max_source_points, seed=seed)
    raise ValueError(f"Unknown pattern mode: {pattern}")


def generate_dataset(
    output_dir: str | Path,
    pattern: str = "text",
    text: str = "ABBE",
    logo_path: str = "",
    dot_mode: str = "single",
    sigma_px: float = DEFAULT_TEXT_SIGMA_PX,
    frames: int = DEFAULT_TEXT_FRAMES,
    blinkers_per_frame: int = DEFAULT_BLINKERS_PER_FRAME,
    spacing_px: float = DEFAULT_TEXT_SPACING_PX,
    photons: float = DEFAULT_TEXT_PHOTONS,
    background: float = DEFAULT_TEXT_BACKGROUND,
    read_noise: float = DEFAULT_TEXT_READ_NOISE,
    fps: float = 12.0,
    seed: int = 2026,
    logo_scale: float = DEFAULT_LOGO_SCALE,
    logo_sampling: str = DEFAULT_LOGO_SAMPLING,
    logo_grid_step_px: int = DEFAULT_LOGO_GRID_STEP_PX,
    logo_max_source_points: int = 0,
    blinker_spacing_mode: str = "separated",
    min_blinker_distance_px: float = DEFAULT_MANUAL_MIN_DISTANCE_PX,
    worker_count: int = 0,
    frame_backend: str = "auto_cpu",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask = build_source_mask(
        pattern,
        text,
        logo_path,
        dot_mode,
        logo_scale=logo_scale,
        logo_sampling=logo_sampling,
        logo_grid_step_px=logo_grid_step_px,
        logo_max_source_points=logo_max_source_points,
        seed=seed,
    )
    edge_padding_px = compute_edge_padding_px(sigma_px)
    points, canvas_shape = mask_to_camera_points(mask, spacing_px=spacing_px, padding_px=edge_padding_px)
    if not points:
        raise ValueError("source mask contains no active points")

    rng = np.random.default_rng(seed)
    frame_jobs = []
    mode = "single" if blinkers_per_frame <= 1 else "multiple"
    frame_min_distance_px = blinker_min_distance_px(blinker_spacing_mode, sigma_px, min_blinker_distance_px)
    frame_photons = effective_photons_per_blinker(photons, blinkers_per_frame)

    for frame_index in range(int(frames)):
        active = choose_active_points(
            points,
            rng,
            mode=mode,
            blinkers_per_frame=blinkers_per_frame,
            spacing_mode=blinker_spacing_mode,
            min_distance_px=frame_min_distance_px,
        )
        frame_jobs.append(
            {
                "frame_index": frame_index,
                "active": active,
                "canvas_shape": canvas_shape,
                "sigma_px": sigma_px,
                "frame_photons": frame_photons,
                "background": background,
                "read_noise": read_noise,
                "render_seed": int(rng.integers(0, 2**31 - 1)),
                "pattern": pattern,
            }
        )

    backend_used, workers, backend_note, gpu_device = resolve_virtual_frame_backend(frame_backend, worker_count, len(frame_jobs))
    if backend_used == "gpu_cuda":
        rendered_frames = [_render_virtual_frame_job_gpu_cuda(job) for job in frame_jobs]
    elif workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rendered_frames = list(executor.map(_render_virtual_frame_job, frame_jobs))
    else:
        rendered_frames = [_render_virtual_frame_job(job) for job in frame_jobs]

    active_counts = [int(item["active_count"]) for item in rendered_frames]
    truth_rows = [row for item in rendered_frames for row in item["truth_rows"]]
    video_frames = [item["video_frame"] for item in rendered_frames]
    preview_frame = next((item["preview_frame"] for item in rendered_frames if item["preview_frame"] is not None), None)

    video_path = write_mp4(out / "blinking_video.mp4", video_frames, fps=fps)
    write_truth_csv(out / "truth.csv", truth_rows)
    Image.fromarray(normalize_to_uint8(preview_frame)).save(out / "preview_frame.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(out / "pattern_mask.png")

    metadata = {
        "generator_name": APP_NAME,
        "generator_version": APP_VERSION,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "pattern_mode": pattern,
        "dot_mode": dot_mode,
        "text": text if pattern == "text" else "",
        "text_mask_mode": TEXT_MASK_MODE if pattern == "text" else "",
        "text_letter_cell_size_px": CELL_SIZE if pattern == "text" else "",
        "text_mask_shape": f"{mask.shape[0]}x{mask.shape[1]}" if pattern == "text" else "",
        "logo_scale": float(logo_scale) if pattern == "logo" else "",
        "logo_sampling": logo_sampling if pattern == "logo" else "",
        "logo_grid_step_px": int(logo_grid_step_px) if pattern == "logo" else "",
        "logo_max_source_points": int(logo_max_source_points) if pattern == "logo" else "",
        "source_point_count": int(mask.sum()),
        "source_mask_shape": f"{mask.shape[0]}x{mask.shape[1]}",
        "image_width_px": int(canvas_shape[1]),
        "image_height_px": int(canvas_shape[0]),
        "frames": int(frames),
        "sigma_px": float(sigma_px),
        "brightness": float(photons),
        "base_brightness_per_blinker": float(photons),
        "effective_brightness_per_blinker": float(frame_photons),
        "brightness_scale_from_max_blinkers": float(frame_photons / float(photons)) if float(photons) else 0.0,
        "background": float(background),
        "read_noise": float(read_noise),
        "blinking_mode": mode,
        "requested_blinkers_per_frame": int(blinkers_per_frame),
        "approximate_blinkers_per_frame": int(blinkers_per_frame),
        "actual_mean_blinkers_per_frame": float(sum(active_counts) / len(active_counts)) if active_counts else 0.0,
        "actual_min_blinkers_per_frame": int(min(active_counts)) if active_counts else 0,
        "actual_max_blinkers_per_frame": int(max(active_counts)) if active_counts else 0,
        "blinker_spacing_mode": blinker_spacing_mode,
        "manual_min_distance_px": float(min_blinker_distance_px),
        "blinker_spacing_multiplier": float(blinker_spacing_multiplier(blinker_spacing_mode)),
        "min_blinker_distance_px": float(frame_min_distance_px),
        "random_seed": int(seed),
        "frame_generation_backend_requested": str(frame_backend or "auto_cpu").lower().replace(" ", "_").replace("-", "_"),
        "frame_generation_backend_used": backend_used,
        "frame_generation_worker_count": int(workers if backend_used in {"cpu_parallel", "serial"} else 1),
        "edge_padding_px": int(edge_padding_px),
        "output_video": str(video_path.name),
        "truth_file": "truth.csv",
        "pattern_mask_file": "pattern_mask.png",
        "notes": "Truth coordinates are padded camera-frame coordinates. Reconstruction must not use truth before running.",
    }
    if backend_note:
        metadata["frame_generation_backend_note"] = backend_note
    if gpu_device:
        metadata["frame_generation_gpu_device"] = gpu_device
    write_metadata(out / "metadata.json", metadata)
    return metadata


def run_gui() -> int:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except Exception as exc:
        raise RuntimeError("PySide6 is not installed. Install PySide6 to use the v7 desktop GUI.") from exc

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(APP_NAME)
            self._add_studio_navigation()
            root = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(root)

            self.steps = QtWidgets.QListWidget()
            self.workflow_rows = [
                {"page": 0, "section": None},
                {"page": 1, "section": None},
                {"page": 2, "section": None},
                {"page": 3, "section": None},
                {"page": 4, "section": "source"},
                {"page": 4, "section": "source"},
                {"page": 4, "section": "sample"},
                {"page": 4, "section": "files"},
            ]
            self.steps.addItems(
                [
                    "1 Pattern",
                    "2 Blinking",
                    "3 Image Formation",
                    "4 Preview",
                    "5 Export",
                    "   Source / Mask",
                    "   Sample Frames",
                    "   Generated Files",
                ]
            )
            for row in range(5, self.steps.count()):
                self.steps.item(row).setForeground(QtGui.QColor("#4b5563"))
            self.steps.setFixedWidth(170)
            layout.addWidget(self.steps)

            center = QtWidgets.QWidget()
            center_layout = QtWidgets.QVBoxLayout(center)
            self.preview_title = QtWidgets.QLabel("Virtual Experiment Preview")
            self.preview_title.setStyleSheet("font-size: 18px; font-weight: 600;")
            self.output_tabs = QtWidgets.QTabBar()
            self.output_tabs.setExpanding(False)
            self.output_tabs.setVisible(False)
            self.output_tabs.currentChanged.connect(self.output_tab_changed)
            self.preview = QtWidgets.QLabel("Choose a pattern, preview it, then export a blinking video.")
            self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.preview.setMinimumSize(620, 360)
            self.preview.setStyleSheet("background: #111; color: #ddd; border: 1px solid #888;")
            self.video_controls = QtWidgets.QWidget()
            video_controls_layout = QtWidgets.QHBoxLayout(self.video_controls)
            video_controls_layout.setContentsMargins(0, 0, 0, 0)
            self.play_video_button = QtWidgets.QPushButton("Play")
            self.play_video_button.clicked.connect(self.toggle_video_playback)
            self.video_frame_label = QtWidgets.QLabel("Frame 0 / 0")
            video_controls_layout.addWidget(self.play_video_button)
            video_controls_layout.addWidget(self.video_frame_label)
            video_controls_layout.addStretch(1)
            self.video_controls.setVisible(False)
            self.log = QtWidgets.QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(150)
            self.log.setText("Ready.\nThe reconstruction app should load the generated MP4, not truth.csv, until comparison.")
            center_layout.addWidget(self.preview_title)
            center_layout.addWidget(self.output_tabs)
            center_layout.addWidget(self.preview, 1)
            center_layout.addWidget(self.video_controls)
            center_layout.addWidget(self.log)
            layout.addWidget(center, 1)

            self.pattern = QtWidgets.QComboBox()
            self.pattern.addItems(["text", "dots", "logo"])
            self.text = QtWidgets.QLineEdit("ABBE")
            self.dot_mode = QtWidgets.QComboBox()
            self.dot_mode.addItems(["single", "pair", "random"])
            self.logo = QtWidgets.QLineEdit("")
            self.logo_scale = QtWidgets.QDoubleSpinBox()
            self.logo_scale.setRange(0.1, 5.0)
            self.logo_scale.setSingleStep(0.1)
            self.logo_scale.setValue(DEFAULT_LOGO_SCALE)
            self.logo_sampling = QtWidgets.QComboBox()
            self.logo_sampling.addItems(["grid", "all_pixels", "random_max_points"])
            self.logo_sampling.setCurrentText(DEFAULT_LOGO_SAMPLING)
            self.logo_grid_step = QtWidgets.QSpinBox()
            self.logo_grid_step.setRange(1, 100)
            self.logo_grid_step.setValue(DEFAULT_LOGO_GRID_STEP_PX)
            self.logo_max_points = QtWidgets.QSpinBox()
            self.logo_max_points.setRange(0, 1000000)
            self.logo_max_points.setValue(0)
            self.session_timestamp = current_timestamp()
            self.output_manually_selected = False
            self.out = QtWidgets.QLineEdit(str(virtual_default_output_dir("text", "ABBE", self.session_timestamp)))
            self.sigma = QtWidgets.QDoubleSpinBox()
            self.sigma.setRange(0.5, 50)
            self.sigma.setValue(DEFAULT_TEXT_SIGMA_PX)
            self.spacing = QtWidgets.QDoubleSpinBox()
            self.spacing.setRange(1.0, 100.0)
            self.spacing.setValue(DEFAULT_TEXT_SPACING_PX)
            self.photons = QtWidgets.QDoubleSpinBox()
            self.photons.setRange(1.0, 1_000_000.0)
            self.photons.setValue(DEFAULT_TEXT_PHOTONS)
            self.background = QtWidgets.QDoubleSpinBox()
            self.background.setRange(0.0, 100_000.0)
            self.background.setValue(DEFAULT_TEXT_BACKGROUND)
            self.read_noise = QtWidgets.QDoubleSpinBox()
            self.read_noise.setRange(0.0, 10_000.0)
            self.read_noise.setValue(DEFAULT_TEXT_READ_NOISE)
            self.frames = QtWidgets.QSpinBox()
            self.frames.setRange(1, 10000)
            self.frames.setValue(DEFAULT_TEXT_FRAMES)
            self.blinkers = QtWidgets.QSpinBox()
            self.blinkers.setRange(1, 1000)
            self.blinkers.setValue(DEFAULT_BLINKERS_PER_FRAME)
            self.cpu_workers = QtWidgets.QSpinBox()
            self.cpu_workers.setRange(0, 64)
            self.cpu_workers.setValue(0)
            self.cpu_workers.setToolTip("CPU workers for frame rendering. 0 = auto, up to 90% of available CPUs; 1 = serial.")
            self.frame_backend = QtWidgets.QComboBox()
            self.frame_backend.addItem("Auto CPU", "auto_cpu")
            self.frame_backend.addItem("CPU parallel", "cpu_parallel")
            self.frame_backend.addItem("GPU CUDA", "gpu_cuda")
            self.frame_backend.setToolTip("Experimental frame renderer. GPU CUDA requires PyTorch CUDA and falls back to CPU if unavailable.")
            self.blinker_spacing = QtWidgets.QComboBox()
            self.blinker_spacing.addItems(["separated", "touching", "overlapping", "random"])
            self.manual_min_distance = QtWidgets.QDoubleSpinBox()
            self.manual_min_distance.setRange(0.0, 10000.0)
            self.manual_min_distance.setDecimals(2)
            self.manual_min_distance.setValue(DEFAULT_MANUAL_MIN_DISTANCE_PX)
            self.effective_distance_label = QtWidgets.QLabel("")
            self.effective_brightness_label = QtWidgets.QLabel("")
            export_button = QtWidgets.QPushButton("Export Dataset")
            export_button.clicked.connect(self.export_dataset)
            self.status = QtWidgets.QLabel("Ready")

            self.current_generation_ready = False
            self.active_output_section = "source"
            self.allow_previous_outputs = QtWidgets.QCheckBox("Allow loading previous generated data")
            self.allow_previous_outputs.setChecked(False)
            self.allow_previous_outputs.toggled.connect(self.previous_outputs_permission_changed)
            self.output_sections = virtual_result_catalog(self.out.text(), include_existing=False)
            self.output_tab_items = []
            self.updating_output_tabs = False
            self.video_frames = []
            self.video_frame_index = 0
            self.video_timer = QtCore.QTimer(self)
            self.video_timer.timeout.connect(self.advance_video_frame)

            self.pages = QtWidgets.QStackedWidget()
            self.pages.addWidget(self._pattern_page())
            self.pages.addWidget(self._blinking_page())
            self.pages.addWidget(self._formation_page())
            self.pages.addWidget(self._preview_page())
            self.pages.addWidget(self._export_page(export_button))
            self.steps.currentRowChanged.connect(self.workflow_changed)
            self.steps.setCurrentRow(0)

            panel_box = QtWidgets.QVBoxLayout()
            title = QtWidgets.QLabel("Workflow Controls")
            title.setStyleSheet("font-size: 18px; font-weight: 600;")
            panel_box.addWidget(title)
            panel_box.addWidget(self.pages, 1)
            panel_box.addWidget(QtWidgets.QLabel("Status"))
            panel_box.addWidget(self.status)
            credit = QtWidgets.QLabel(f"{APP_NAME} {APP_VERSION}\n{AUTHOR_CREDIT}\n{AUTHOR_AFFILIATION}")
            credit.setWordWrap(True)
            credit.setStyleSheet("color: #6b7280; font-size: 10px; padding-top: 8px;")
            panel_box.addWidget(credit)
            side = QtWidgets.QWidget()
            side.setLayout(panel_box)
            side.setFixedWidth(380)
            layout.addWidget(side)

            self.setCentralWidget(root)
            self.resize(1240, 760)
            self.pattern.currentTextChanged.connect(self.update_suggested_output_folder)
            self.text.textChanged.connect(self.update_suggested_output_folder)
            self.dot_mode.currentTextChanged.connect(self.update_suggested_output_folder)
            self.blinker_spacing.currentTextChanged.connect(self.update_derived_settings)
            self.manual_min_distance.valueChanged.connect(self.update_derived_settings)
            self.blinkers.valueChanged.connect(self.update_derived_settings)
            self.photons.valueChanged.connect(self.update_derived_settings)
            self.update_derived_settings()

        def _add_studio_navigation(self):
            toolbar = QtWidgets.QToolBar("Studio")
            toolbar.setMovable(False)
            self.back_to_studio_button = QtWidgets.QPushButton("Back to Studio")
            self.back_to_studio_button.setToolTip("Return to the ClassRoomSTORM Studio launcher")
            self.back_to_studio_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.back_to_studio_button.setStyleSheet(
                """
                QPushButton {
                    background: #1d4ed8;
                    color: white;
                    border: 1px solid #1e40af;
                    border-radius: 6px;
                    font-weight: 700;
                    padding: 7px 14px;
                }
                QPushButton:hover {
                    background: #2563eb;
                    border-color: #1d4ed8;
                }
                QPushButton:pressed {
                    background: #1e3a8a;
                }
                """
            )
            self.back_to_studio_button.clicked.connect(self.back_to_studio)
            toolbar.addWidget(self.back_to_studio_button)
            self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)

        def back_to_studio(self):
            script = studio_launcher_script()
            if not script.exists():
                QtWidgets.QMessageBox.warning(self, APP_NAME, f"Studio launcher not found:\n{script}")
                return
            subprocess.Popen([sys.executable, str(script)], cwd=str(PROJECT_ROOT))
            QtWidgets.QApplication.instance().quit()

        def _pattern_page(self):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            logo_browse = QtWidgets.QPushButton("Browse Logo")
            logo_browse.clicked.connect(self.browse_logo)
            logo_row = QtWidgets.QWidget()
            logo_layout = QtWidgets.QHBoxLayout(logo_row)
            logo_layout.setContentsMargins(0, 0, 0, 0)
            logo_layout.addWidget(self.logo, 1)
            logo_layout.addWidget(logo_browse)
            form.addRow("Pattern", self.pattern)
            form.addRow("Text", self.text)
            form.addRow("Dot mode", self.dot_mode)
            form.addRow("Logo", logo_row)
            form.addRow("Logo scale", self.logo_scale)
            form.addRow("Logo sampling", self.logo_sampling)
            form.addRow("Logo grid step px", self.logo_grid_step)
            form.addRow("Logo max source points", self.logo_max_points)
            note = QtWidgets.QLabel("Text uses the fixed 16x16 LED alphabet. Logo mode can be scaled and sampled before export; preview the source mask before generating.")
            note.setWordWrap(True)
            form.addRow(note)
            return page

        def _blinking_page(self):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            form.addRow("Frames", self.frames)
            form.addRow("Max blinkers/frame", self.blinkers)
            form.addRow("Blinker spacing", self.blinker_spacing)
            form.addRow("Manual min distance px", self.manual_min_distance)
            form.addRow("Effective min distance", self.effective_distance_label)
            note = QtWidgets.QLabel("Max blinkers/frame is a maximum. Separated/touching/overlapping modes multiply the manual distance; random uses no spacing control.")
            note.setWordWrap(True)
            form.addRow(note)
            return page

        def _formation_page(self):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            form.addRow("sigma_px", self.sigma)
            form.addRow("Spacing px", self.spacing)
            form.addRow("Brightness", self.photons)
            form.addRow("Effective brightness", self.effective_brightness_label)
            form.addRow("Background", self.background)
            form.addRow("Read noise", self.read_noise)
            return page

        def _preview_page(self):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            mask_button = QtWidgets.QPushButton("Preview Source Mask")
            mask_button.clicked.connect(self.preview_mask)
            frame_button = QtWidgets.QPushButton("Preview Sample Blink Frame")
            frame_button.clicked.connect(self.preview_sample_frame)
            form.addRow(mask_button)
            form.addRow(frame_button)
            note = QtWidgets.QLabel("Preview does not write files. Export creates the MP4, truth CSV, metadata JSON, and preview images.")
            note.setWordWrap(True)
            form.addRow(note)
            return page

        def _export_page(self, export_button):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            browse_output = QtWidgets.QPushButton("Browse Output Folder")
            browse_output.clicked.connect(self.browse_output_folder)
            output_row = QtWidgets.QWidget()
            output_layout = QtWidgets.QHBoxLayout(output_row)
            output_layout.setContentsMargins(0, 0, 0, 0)
            output_layout.addWidget(self.out, 1)
            output_layout.addWidget(browse_output)
            form.addRow("Output", output_row)
            form.addRow("Frame backend", self.frame_backend)
            form.addRow("CPU workers (0=auto)", self.cpu_workers)
            form.addRow(self.allow_previous_outputs)
            form.addRow(export_button)
            load_previous = QtWidgets.QPushButton("Load Previous Generated Folder")
            load_previous.clicked.connect(self.browse_previous_output_folder)
            refresh = QtWidgets.QPushButton("Refresh Output Browser")
            refresh.clicked.connect(lambda: self.refresh_output_browser(show_first=True))
            self.load_previous_outputs_button = load_previous
            self.load_previous_outputs_button.setEnabled(False)
            buttons = QtWidgets.QWidget()
            buttons_layout = QtWidgets.QHBoxLayout(buttons)
            buttons_layout.setContentsMargins(0, 0, 0, 0)
            buttons_layout.addWidget(load_previous)
            buttons_layout.addWidget(refresh)
            form.addRow(buttons)
            return page

        @QtCore.Slot(int)
        def workflow_changed(self, row):
            if row < 0 or row >= len(self.workflow_rows):
                return
            target = self.workflow_rows[row]
            self.pages.setCurrentIndex(int(target["page"]))
            section = target["section"]
            if section is None:
                self.output_tabs.setVisible(False)
                return
            self.active_output_section = str(section)
            self.refresh_output_browser(show_first=True)

        def current_mask(self):
            return build_source_mask(
                self.pattern.currentText(),
                self.text.text(),
                self.logo.text(),
                self.dot_mode.currentText(),
                logo_scale=float(self.logo_scale.value()),
                logo_sampling=self.logo_sampling.currentText(),
                logo_grid_step_px=int(self.logo_grid_step.value()),
                logo_max_source_points=int(self.logo_max_points.value()),
            )

        def current_min_distance_px(self):
            return blinker_min_distance_px(
                self.blinker_spacing.currentText(),
                float(self.sigma.value()),
                float(self.manual_min_distance.value()),
            )

        def current_effective_photons(self):
            return effective_photons_per_blinker(float(self.photons.value()), int(self.blinkers.value()))

        def update_derived_settings(self):
            distance = self.current_min_distance_px()
            multiplier = blinker_spacing_multiplier(self.blinker_spacing.currentText())
            if multiplier <= 0:
                distance_text = "0 px (random)"
            else:
                distance_text = f"{distance:.1f} px ({multiplier:g}x manual)"
            self.effective_distance_label.setText(distance_text)
            self.effective_brightness_label.setText(
                f"{self.current_effective_photons():.0f} per blinker"
            )

        def current_sample_frame(self):
            mask = self.current_mask()
            padding = compute_edge_padding_px(float(self.sigma.value()))
            points, canvas_shape = mask_to_camera_points(mask, spacing_px=float(self.spacing.value()), padding_px=padding)
            if not points:
                raise ValueError("source mask contains no active points")
            rng = np.random.default_rng(2026)
            mode = "single" if int(self.blinkers.value()) <= 1 else "multiple"
            active = choose_active_points(
                points,
                rng,
                mode=mode,
                blinkers_per_frame=int(self.blinkers.value()),
                spacing_mode=self.blinker_spacing.currentText(),
                min_distance_px=self.current_min_distance_px(),
            )
            return render_gaussian_frame(
                active,
                canvas_shape,
                sigma_px=float(self.sigma.value()),
                photons=self.current_effective_photons(),
                background=float(self.background.value()),
                read_noise=float(self.read_noise.value()),
                seed=2026,
            )

        def _show_array(self, array, title):
            arr = normalize_to_uint8(array)
            h, w = arr.shape
            qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8).copy()
            pix = QtGui.QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                self.preview.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self._reset_image_preview_style()
            self.stop_video_playback()
            self.video_controls.setVisible(False)
            self.preview.setPixmap(scaled)
            self.preview_title.setText(title)

        def _show_image_file(self, path, title):
            pix = QtGui.QPixmap(str(path))
            if pix.isNull():
                self.log.append(f"Could not preview image: {path}")
                return
            scaled = pix.scaled(
                self.preview.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self._reset_image_preview_style()
            self.stop_video_playback()
            self.video_controls.setVisible(False)
            self.preview.setPixmap(scaled)
            self.preview_title.setText(title)

        def _show_text_file(self, path, title):
            self.stop_video_playback()
            self.video_controls.setVisible(False)
            try:
                text = Path(path).read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                self.log.append(f"Could not preview text file: {exc}")
                return
            lines = text.splitlines()
            if len(lines) > 80:
                text = "\n".join(lines[:80]) + f"\n\n... showing first 80 lines of {len(lines)}"
            self.preview.clear()
            self.preview.setText(text or "(empty file)")
            self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            self.preview.setStyleSheet("background: #111; color: #eee; border: 1px solid #888; padding: 12px; font-family: Consolas, monospace;")
            self.preview_title.setText(title)

        def _show_file_info(self, path, title):
            self.stop_video_playback()
            self.video_controls.setVisible(False)
            self.preview.clear()
            self.preview.setText(f"{title}\n\n{Path(path).resolve()}")
            self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.preview.setStyleSheet("background: #111; color: #eee; border: 1px solid #888; padding: 12px;")
            self.preview_title.setText(title)

        def _show_video_file(self, path, title):
            try:
                self.video_frames = read_video_frames(path)
            except Exception as exc:
                self._show_file_info(path, title)
                self.status.setText(f"Could not preview video: {exc}")
                return
            if not self.video_frames:
                self._show_file_info(path, title)
                self.status.setText("Video contains no readable frames")
                return
            self.video_frame_index = 0
            self.preview_title.setText(title)
            self.video_controls.setVisible(True)
            self.play_video_button.setText("Play")
            self.show_video_frame()

        def show_video_frame(self):
            if not self.video_frames:
                return
            self._reset_image_preview_style()
            frame = self.video_frames[self.video_frame_index]
            arr = normalize_to_uint8(frame)
            h, w = arr.shape
            qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8).copy()
            pix = QtGui.QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                self.preview.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.preview.setPixmap(scaled)
            self.video_frame_label.setText(f"Frame {self.video_frame_index + 1} / {len(self.video_frames)}")

        @QtCore.Slot()
        def toggle_video_playback(self):
            if not self.video_frames:
                return
            if self.video_timer.isActive():
                self.video_timer.stop()
                self.play_video_button.setText("Play")
            else:
                self.video_timer.start(120)
                self.play_video_button.setText("Pause")

        @QtCore.Slot()
        def advance_video_frame(self):
            if not self.video_frames:
                self.stop_video_playback()
                return
            self.video_frame_index = (self.video_frame_index + 1) % len(self.video_frames)
            self.show_video_frame()

        def stop_video_playback(self):
            if self.video_timer.isActive():
                self.video_timer.stop()
            if hasattr(self, "play_video_button"):
                self.play_video_button.setText("Play")

        def _reset_image_preview_style(self):
            self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.preview.setStyleSheet("background: #111; color: #ddd; border: 1px solid #888;")

        @QtCore.Slot()
        def browse_logo(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose logo image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)")
            if path:
                self.logo.setText(path)
                self.pattern.setCurrentText("logo")

        @QtCore.Slot()
        def browse_output_folder(self):
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder for this virtual dataset", self.out.text())
            if folder:
                self.output_manually_selected = True
                self.out.setText(folder)
                self.current_generation_ready = False
                self.refresh_output_browser(show_first=False)
                self.status.setText("Output folder selected. Generated files will appear after export.")

        @QtCore.Slot()
        def browse_previous_output_folder(self):
            if not self.allow_previous_outputs.isChecked():
                self.status.setText("Check 'Allow loading previous generated data' before opening old folders.")
                return
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose previous virtual output folder", self.out.text())
            if folder:
                self.output_manually_selected = True
                self.out.setText(folder)
                self.current_generation_ready = False
                self.refresh_output_browser(show_first=True)

        @QtCore.Slot(bool)
        def previous_outputs_permission_changed(self, checked):
            self.load_previous_outputs_button.setEnabled(bool(checked))
            if not checked and not self.current_generation_ready:
                self.refresh_output_browser(show_first=False)
                self.status.setText("Previous generated data hidden.")

        @QtCore.Slot()
        def preview_mask(self):
            try:
                mask = self.current_mask()
                self._show_array(mask * 255, "Source Mask Preview")
                self.status.setText(f"Mask points: {int(mask.sum())}")
            except Exception as exc:
                self.status.setText(str(exc))
                self.log.append(f"Preview mask error: {exc}")

        @QtCore.Slot()
        def preview_sample_frame(self):
            try:
                frame = self.current_sample_frame()
                padding = compute_edge_padding_px(float(self.sigma.value()))
                self._show_array(frame, "Sample Blink Frame Preview")
                self.status.setText(f"Preview generated with adaptive edge padding: {padding} px")
            except Exception as exc:
                self.status.setText(str(exc))
                self.log.append(f"Preview frame error: {exc}")

        @QtCore.Slot()
        def update_suggested_output_folder(self, *_args):
            if self.output_manually_selected or self.current_generation_ready:
                return
            self.out.setText(str(virtual_default_output_dir(self.pattern.currentText(), self.text.text(), self.session_timestamp)))

        @QtCore.Slot()
        def export_dataset(self):
            try:
                metadata = generate_dataset(
                    self.out.text(),
                    pattern=self.pattern.currentText(),
                    text=self.text.text(),
                    logo_path=self.logo.text(),
                    dot_mode=self.dot_mode.currentText(),
                    sigma_px=float(self.sigma.value()),
                    frames=int(self.frames.value()),
                    blinkers_per_frame=int(self.blinkers.value()),
                    spacing_px=float(self.spacing.value()),
                    photons=float(self.photons.value()),
                    background=float(self.background.value()),
                    read_noise=float(self.read_noise.value()),
                    logo_scale=float(self.logo_scale.value()),
                    logo_sampling=self.logo_sampling.currentText(),
                    logo_grid_step_px=int(self.logo_grid_step.value()),
                    logo_max_source_points=int(self.logo_max_points.value()),
                    blinker_spacing_mode=self.blinker_spacing.currentText(),
                    min_blinker_distance_px=float(self.manual_min_distance.value()),
                    worker_count=int(self.cpu_workers.value()),
                    frame_backend=str(self.frame_backend.currentData() or "auto_cpu"),
                )
                self.current_generation_ready = True
                self.status.setText(f"Saved {metadata['output_video']} in {self.out.text()}")
                self.log.append(f"Saved dataset in {self.out.text()}")
                self.steps.setCurrentRow(4)
                self.active_output_section = "source"
                self.refresh_output_browser(show_first=True)
            except Exception as exc:
                self.status.setText(str(exc))
                self.log.append(f"Export error: {exc}")

        @QtCore.Slot()
        def refresh_output_browser(self, show_first=True):
            include_existing = self.current_generation_ready or self.allow_previous_outputs.isChecked()
            self.output_sections = virtual_result_catalog(self.out.text(), include_existing=include_existing)
            section = next((item for item in self.output_sections if item["key"] == self.active_output_section), None)
            self.updating_output_tabs = True
            while self.output_tabs.count():
                self.output_tabs.removeTab(0)
            self.output_tab_items = []
            if section is not None:
                self.preview_title.setText(section["label"])
                for item in section["items"]:
                    self.output_tabs.addTab(item["label"])
                    self.output_tab_items.append(item)
            self.output_tabs.setVisible(bool(self.output_tab_items))
            self.updating_output_tabs = False
            if not self.output_tab_items:
                self.preview.clear()
                if include_existing:
                    message = f"No generated files found for {section['label'] if section else 'this section'}.\nExport a dataset or choose a previous generated folder."
                else:
                    message = "No generated files loaded yet.\nExport a dataset to show new files, or allow loading previous generated data."
                self.preview.setText(message)
                self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.status.setText("No generated files loaded")
                return
            if show_first:
                self.output_tabs.setCurrentIndex(0)
                self.show_output_item(self.output_tab_items[0])

        @QtCore.Slot(int)
        def output_tab_changed(self, index):
            if self.updating_output_tabs or index < 0 or index >= len(self.output_tab_items):
                return
            self.show_output_item(self.output_tab_items[index])

        def show_output_item(self, item):
            path = item["path"]
            if item["kind"] == "image":
                self._show_image_file(path, item["label"])
            elif item["kind"] == "text":
                self._show_text_file(path, item["label"])
            elif item["kind"] == "video":
                self._show_video_file(path, item["label"])
            else:
                self._show_file_info(path, item["label"])
            self.status.setText(f"Showing {item['label']}")

    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    return app.exec()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--generate-demo", action="store_true", help="Generate a dataset without opening the GUI.")
    parser.add_argument("--out", default="")
    parser.add_argument("--pattern", choices=["dots", "text", "logo"], default="text")
    parser.add_argument("--text", default="ABBE")
    parser.add_argument("--logo", default="")
    parser.add_argument("--dot-mode", choices=["single", "pair", "random"], default="single")
    parser.add_argument("--sigma-px", type=float, default=DEFAULT_TEXT_SIGMA_PX)
    parser.add_argument("--frames", type=int, default=DEFAULT_TEXT_FRAMES)
    parser.add_argument("--blinkers-per-frame", type=int, default=DEFAULT_BLINKERS_PER_FRAME)
    parser.add_argument("--spacing-px", type=float, default=DEFAULT_TEXT_SPACING_PX)
    parser.add_argument("--photons", type=float, default=DEFAULT_TEXT_PHOTONS)
    parser.add_argument("--background", type=float, default=DEFAULT_TEXT_BACKGROUND)
    parser.add_argument("--read-noise", type=float, default=DEFAULT_TEXT_READ_NOISE)
    parser.add_argument("--logo-scale", type=float, default=DEFAULT_LOGO_SCALE)
    parser.add_argument("--logo-sampling", choices=["grid", "all_pixels", "random_max_points"], default=DEFAULT_LOGO_SAMPLING)
    parser.add_argument("--logo-grid-step-px", type=int, default=DEFAULT_LOGO_GRID_STEP_PX)
    parser.add_argument("--logo-max-source-points", type=int, default=0)
    parser.add_argument("--blinker-spacing-mode", choices=["separated", "touching", "overlapping", "random"], default="separated")
    parser.add_argument("--min-blinker-distance-px", type=float, default=DEFAULT_MANUAL_MIN_DISTANCE_PX)
    parser.add_argument("--workers", type=int, default=0, help="CPU workers for virtual frame rendering. Use 0 for auto and 1 for serial.")
    parser.add_argument("--frame-backend", choices=VIRTUAL_FRAME_BACKENDS, default="auto_cpu", help="Virtual frame rendering backend. GPU CUDA is experimental and falls back to CPU if unavailable.")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.generate_demo:
        output_dir = args.out or virtual_default_output_dir(args.pattern, args.text)
        metadata = generate_dataset(
            output_dir=output_dir,
            pattern=args.pattern,
            text=args.text,
            logo_path=args.logo,
            dot_mode=args.dot_mode,
            sigma_px=args.sigma_px,
            frames=args.frames,
            blinkers_per_frame=args.blinkers_per_frame,
            spacing_px=args.spacing_px,
            photons=args.photons,
            background=args.background,
            read_noise=args.read_noise,
            logo_scale=args.logo_scale,
            logo_sampling=args.logo_sampling,
            logo_grid_step_px=args.logo_grid_step_px,
            logo_max_source_points=args.logo_max_source_points,
            blinker_spacing_mode=args.blinker_spacing_mode,
            min_blinker_distance_px=args.min_blinker_distance_px,
            worker_count=args.workers,
            frame_backend=args.frame_backend,
            seed=args.seed,
        )
        print(f"Saved dataset to {output_dir}")
        print(f"Video: {metadata['output_video']}")
        print(f"Edge padding px: {metadata['edge_padding_px']}")
        return 0
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
