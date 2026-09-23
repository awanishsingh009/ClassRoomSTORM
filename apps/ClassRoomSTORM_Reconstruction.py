"""
ClassRoomSTORM Reconstruction V1.2.

This script is the student-facing reconstruction app. It reconstructs from the
video itself first. Virtual truth data, when present, is used only after
reconstruction through a comparison step.
"""

from __future__ import annotations

import argparse
import hashlib
import datetime as _dt
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED = PROJECT_ROOT / "classroomstorm_core"
sys.path.insert(0, str(SHARED))

from plotting import (
    save_cumulative_reconstruction_panel,
    save_density_map,
    save_grayscale_png,
    save_localization_scatter,
    save_raw_localization_overlay,
    save_side_by_side,
    save_superres_png,
)
from pipeline_report import generate_pipeline_report
from profile_tools import linked_line_profiles, linked_profile_series
from reconstruction import normalize_crop, reconstruct_frames, write_localizations_csv, write_summary_json
from truth_comparison import match_localizations_to_truth, read_truth_csv
from version import APP_VERSION
from video_io import read_video_frames, read_video_info


APP_NAME = "ClassRoomSTORM Reconstruction"
AUTHOR_CREDIT = "Developed by Dr. Awanish Pratap Singh"
AUTHOR_AFFILIATION = "Institute of Biomedical Optics, University of Lübeck"

# Frame cap when loading the raw video for in-app playback on the 1 Load Video
# step. Keeps memory bounded for long videos; the sample video is well under it.
RAW_VIDEO_PLAYBACK_CAP = 1500


def processing_backend_options() -> list[str]:
    return ["serial", "auto_cpu", "gpu_only", "auto_cpu_gpu", "auto_cpu_gpu_acceleration"]


def processing_backend_label(backend: str) -> str:
    labels = {
        "serial": "Serial",
        "auto_cpu_gpu": "Auto CPU+GPU",
        "auto_cpu": "Auto CPU",
        "cpu_parallel": "CPU parallel",
        "hybrid_cpu_gpu": "CPU + GPU hybrid",
        "gpu_only": "Only GPU",
        "gpu_experimental": "Only GPU",
        "auto_cpu_gpu_acceleration": "Auto CPU+GPU acceleration",
    }
    return labels.get(backend, str(backend))


def gpu_dependency_guidance(backend: str) -> str:
    key = str(backend or "serial").lower().replace(" ", "_").replace("-", "_")
    if key in {"serial", "auto_cpu"}:
        return (
            "No GPU dependency is needed for this option. Serial and Auto CPU use the standard "
            "ClassRoomSTORM requirements only."
        )
    return (
        "GPU dependency notes:\n"
        "NVIDIA Windows/Linux: install the optional PyTorch CUDA backend with "
        "`python -m pip install -r ClassRoomSTORM_V1/requirements-gpu.txt`.\n"
        "AMD GPU: not currently implemented in this app; use Auto CPU or Auto CPU+GPU fallback.\n"
        "Intel GPU: not currently implemented in this app; use Auto CPU or Auto CPU+GPU fallback.\n"
        "MacBook / Apple Silicon: PyTorch MPS support is possible future work, but not currently "
        "implemented here; use Auto CPU for reliable classroom runs."
    )


def current_timestamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def safe_name_part(text: str, fallback: str = "") -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or fallback


def reconstruction_default_output_dir(label: str = "", timestamp: str | None = None) -> Path:
    stamp = timestamp or current_timestamp()
    cleaned = safe_name_part(label)
    folder = f"ClassRoomSTORM_Reconstruction_{cleaned}_{stamp}" if cleaned else f"ClassRoomSTORM_Reconstruction_{stamp}"
    return PROJECT_ROOT / "results" / "reconstructions" / folder


def studio_launcher_script() -> Path:
    return PROJECT_ROOT / "apps" / "ClassRoomSTORM_Studio.py"


def clip_crop_to_shape(crop: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Clip a requested crop to a frame shape and return None when it is empty."""
    try:
        return normalize_crop(crop, shape)
    except ValueError:
        return None


def clamp_preview_zoom(value: float) -> float:
    """Keep result preview zoom in a classroom-friendly range."""
    return max(0.25, min(4.0, round(float(value), 2)))


def preview_zoom_label(value: float) -> str:
    return f"{int(round(clamp_preview_zoom(value) * 100))}%"


def document_preview_text(path: str | Path, max_lines: int = 5000) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n\n... showing first {max_lines} lines of {len(lines)}"
    return text or "(empty file)"


def result_catalog(output_dir: str | Path, include_existing: bool = True) -> list[dict]:
    """Return existing reconstruction outputs grouped for the student result browser."""
    out = Path(output_dir)
    main_items = [
        ("Raw Mean", "raw_mean.png", "image"),
        ("Super-Resolution", "superres.png", "image"),
        ("Raw vs Super-Resolution", "raw_vs_superres.png", "image"),
    ]
    if (out / "raw_mean.npy").exists() and (out / "superres.npy").exists():
        main_items.append(("Linked Profiles", "raw_mean.npy", "linked_profile"))
    sections = [
        {
            "key": "main",
            "label": "Main Results",
            "items": main_items,
        },
        {
            "key": "diagnostics",
            "label": "Diagnostics",
            "items": [
                ("Localization Scatter", "localization_scatter.png", "image"),
                ("Density Map", "plot_density_map.png", "image"),
                ("Raw Overlay", "plot_raw_localization_overlay.png", "image"),
            ],
        },
        {
            "key": "progress",
            "label": "Learning Progress",
            "items": [
                ("Cumulative 10/30/60/100", "plot_cumulative_reconstruction.png", "image"),
            ],
        },
        {
            "key": "teaching",
            "label": "Teaching Pipeline",
            "items": [
                ("Pipeline Report", "pipeline_report.md", "text"),
                ("Pipeline HTML", "pipeline_report.html", "text"),
                ("Pipeline Flowchart", "pipeline_flowchart.svg", "svg"),
                ("Equation Map", "pipeline_equation_map.svg", "svg"),
            ],
        },
        {
            "key": "data",
            "label": "Data Files",
            "items": [
                ("Summary JSON", "summary.json", "text"),
                ("Localizations CSV", "localizations.csv", "text"),
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


def run_reconstruction(
    video_path: str | Path,
    output_dir: str | Path,
    max_frames: int | None = None,
    threshold_quantile: float = 0.995,
    blinker_mode: str = "single",
    compare_truth: bool = False,
    cumulative_frames: list[int] | None = None,
    crop: tuple[int, int, int, int] | None = None,
    processing_backend: str = "serial",
    worker_count: int | None = None,
    pixel_size: float | None = None,
    unit: str = "px",
    background_mode: str = "none",
    truth_radius_px: float = 2.0,
) -> dict:
    video = Path(video_path)
    out = Path(output_dir)
    if not video.is_file():
        raise ValueError(f"Video not found: {video}")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output folder is not empty. Choose a new folder to preserve previous results.")
    if not np.isfinite(truth_radius_px) or truth_radius_px <= 0:
        raise ValueError("Truth matching radius must be finite and positive")
    frames = read_video_frames(video, max_frames=max_frames)
    result = reconstruct_frames(
        frames,
        threshold_quantile=threshold_quantile,
        blinker_mode=blinker_mode,
        crop=crop,
        processing_backend=processing_backend,
        worker_count=worker_count,
        pixel_size=pixel_size,
        unit=unit,
        background_mode=background_mode,
    )

    out.mkdir(parents=True, exist_ok=True)
    write_localizations_csv(out / "localizations.csv", result.localizations)
    np.save(out / "raw_mean.npy", result.raw_mean)
    np.save(out / "superres.npy", result.superres)
    save_grayscale_png(out / "raw_mean.png", result.raw_mean)
    save_superres_png(out / "superres.png", result.superres)
    save_side_by_side(out / "raw_vs_superres.png", result.raw_mean, result.superres, "Raw (mean)", "Super-Resolution")
    save_localization_scatter(out / "localization_scatter.png", result.localizations)
    save_density_map(out / "plot_density_map.png", result.localizations, result.raw_mean.shape)
    save_raw_localization_overlay(out / "plot_raw_localization_overlay.png", result.raw_mean, result.localizations)
    save_cumulative_reconstruction_panel(
        out / "plot_cumulative_reconstruction.png",
        result.localizations,
        result.raw_mean.shape,
        manual_checkpoints=cumulative_frames,
        total_frames=int(result.summary.get("frames_processed", len(frames))),
    )
    summary = dict(result.summary)
    summary.update(
        {
            "software": f"{APP_NAME} {APP_VERSION}",
            "video_name": video.name,
            "video_path": str(video),
            "pixel_mode": "calibrated" if pixel_size is not None else "pixels",
            "render_kernel": "Fractional-coordinate Gaussian visualization (unit integral)",
            "cumulative_frames": cumulative_frames or "auto:10%,30%,60%,100%",
            "crop_enabled": crop is not None,
            "truth_used_for_reconstruction": False,
        }
    )
    with video.open("rb") as source:
        summary["input_sha256"] = hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _sha256_stream(source)

    truth_path = video.parent / "truth.csv"
    if compare_truth and truth_path.exists():
        truth_rows = read_truth_csv(truth_path)
        summary["truth_comparison"] = match_localizations_to_truth(
            result.localizations, truth_rows, truth_radius_px,
            frames_processed=summary["frames_processed"], crop=summary.get("crop"),
        )
        summary["truth_comparison"]["truth_file"] = str(truth_path)
    elif compare_truth:
        summary["truth_comparison_note"] = "No truth.csv exists next to the input video."
    generate_pipeline_report(out, summary)
    write_summary_json(out / "summary.json", summary)
    return summary


def _sha256_stream(source) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def run_gui() -> int:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except Exception as exc:
        raise RuntimeError("PySide6 is not installed. Install PySide6 to use the v7 desktop GUI.") from exc
    from simulation import normalize_to_uint8

    class ReconstructionWorker(QtCore.QThread):
        completed = QtCore.Signal(dict)
        failed = QtCore.Signal(str)

        def __init__(self, kwargs, parent):
            super().__init__(parent)
            self.kwargs = kwargs

        def run(self):
            try:
                self.completed.emit(run_reconstruction(**self.kwargs))
            except Exception as exc:
                self.failed.emit(str(exc))

    class LinkedProfileWidget(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.raw = None
            self.recon = None
            self.selected_x = 0
            self.selected_y = 0
            self.raw_rect = QtCore.QRectF()
            self.recon_rect = QtCore.QRectF()
            self.setMinimumSize(720, 620)
            self.setMouseTracking(True)

        def set_images(self, raw_image, reconstruction):
            self.raw = np.asarray(raw_image, dtype=float)
            self.recon = np.asarray(reconstruction, dtype=float)
            if self.raw.ndim != 2 or self.recon.ndim != 2:
                raise ValueError("Linked profile view requires 2D raw and reconstruction arrays.")
            self.selected_x = max(0, int(self.raw.shape[1] // 2))
            self.selected_y = max(0, int(self.raw.shape[0] // 2))
            self.update()

        def _fit_rect(self, area, image_shape):
            img_h, img_w = [max(1, int(v)) for v in image_shape[:2]]
            scale = min(area.width() / img_w, area.height() / img_h)
            width = img_w * scale
            height = img_h * scale
            return QtCore.QRectF(
                area.x() + (area.width() - width) / 2,
                area.y() + (area.height() - height) / 2,
                width,
                height,
            )

        def _layout_rects(self):
            width = max(1, self.width())
            height = max(1, self.height())
            margin = 24.0
            gap = 24.0
            title_h = 24.0
            image_h = max(180.0, height * 0.46)
            half_w = (width - margin * 2 - gap) / 2
            raw_area = QtCore.QRectF(margin, margin + title_h, half_w, image_h - title_h)
            recon_area = QtCore.QRectF(margin + half_w + gap, margin + title_h, half_w, image_h - title_h)
            self.raw_rect = self._fit_rect(raw_area, self.raw.shape)
            self.recon_rect = self._fit_rect(recon_area, self.recon.shape)
            profile_top = margin + image_h + 28.0
            profile_h = max(90.0, (height - profile_top - margin - gap) / 2)
            x_profile = QtCore.QRectF(margin, profile_top, width - margin * 2, profile_h)
            y_profile = QtCore.QRectF(margin, profile_top + profile_h + gap, width - margin * 2, profile_h)
            return raw_area, recon_area, x_profile, y_profile

        def _image_pixmap(self, image):
            arr = np.ascontiguousarray(normalize_to_uint8(image))
            h, w = arr.shape
            qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8).copy()
            return QtGui.QPixmap.fromImage(qimg)

        def _draw_image_panel(self, painter, area, image_rect, image, title, selected_shape):
            painter.setPen(QtGui.QColor("#111827"))
            painter.drawText(QtCore.QRectF(area.x(), area.y() - 22, area.width(), 20), int(QtCore.Qt.AlignmentFlag.AlignCenter), title)
            pix = self._image_pixmap(image)
            painter.drawPixmap(image_rect, pix, QtCore.QRectF(pix.rect()))
            painter.setPen(QtGui.QPen(QtGui.QColor("#334155"), 1))
            painter.drawRect(image_rect)

            img_h, img_w = [max(1, int(v)) for v in selected_shape[:2]]
            panel_x = max(0, min(img_w - 1, int(self.selected_x)))
            panel_y = max(0, min(img_h - 1, int(self.selected_y)))
            cross_x = image_rect.left() + (panel_x / max(1, img_w - 1)) * image_rect.width()
            cross_y = image_rect.top() + (panel_y / max(1, img_h - 1)) * image_rect.height()
            pen = QtGui.QPen(QtGui.QColor("#22d3ee"), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(cross_x, image_rect.top()), QtCore.QPointF(cross_x, image_rect.bottom()))
            painter.drawLine(QtCore.QPointF(image_rect.left(), cross_y), QtCore.QPointF(image_rect.right(), cross_y))

        def _profile_points(self, rect, profile):
            values = np.asarray(profile, dtype=float)
            if values.size == 0:
                return []
            points = []
            for idx, value in enumerate(values):
                x = rect.left() + (idx / max(1, values.size - 1)) * rect.width()
                y = rect.bottom() - float(value) * rect.height()
                points.append(QtCore.QPointF(x, y))
            return points

        def _draw_profile_panel(self, painter, rect, title, raw_profile, recon_profile):
            series_styles = linked_profile_series()
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1"), 1))
            painter.setBrush(QtGui.QColor("#ffffff"))
            painter.drawRect(rect)
            painter.setPen(QtGui.QColor("#111827"))
            painter.drawText(
                QtCore.QRectF(rect.x(), rect.y() - 22, rect.width(), 18),
                int(QtCore.Qt.AlignmentFlag.AlignLeft),
                title,
            )

            plot_rect = rect.adjusted(44, 18, -16, -34)
            painter.setPen(QtGui.QPen(QtGui.QColor("#111827"), 1))
            painter.drawLine(QtCore.QPointF(plot_rect.left(), plot_rect.bottom()), QtCore.QPointF(plot_rect.right(), plot_rect.bottom()))
            painter.drawLine(QtCore.QPointF(plot_rect.left(), plot_rect.top()), QtCore.QPointF(plot_rect.left(), plot_rect.bottom()))

            for frac in (0.25, 0.5, 0.75):
                y = plot_rect.bottom() - plot_rect.height() * frac
                painter.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb"), 1))
                painter.drawLine(QtCore.QPointF(plot_rect.left(), y), QtCore.QPointF(plot_rect.right(), y))

            series = [
                (raw_profile, series_styles[0]),
                (recon_profile, series_styles[1]),
            ]
            for profile, style in series:
                color = QtGui.QColor(style["color"])
                points = self._profile_points(plot_rect, profile)
                if len(points) > 1:
                    pen = QtGui.QPen(color, 2)
                    pen.setCosmetic(True)
                    painter.setPen(pen)
                    painter.drawPolyline(points)

            painter.setPen(QtGui.QColor("#475569"))
            painter.drawText(
                QtCore.QRectF(plot_rect.left(), plot_rect.bottom() + 8, plot_rect.width(), 18),
                int(QtCore.Qt.AlignmentFlag.AlignCenter),
                "pixel coordinate",
            )
            painter.save()
            painter.translate(rect.left() + 14, plot_rect.center().y())
            painter.rotate(-90)
            painter.drawText(QtCore.QRectF(-60, -8, 120, 16), int(QtCore.Qt.AlignmentFlag.AlignCenter), "normalized intensity")
            painter.restore()

            legend_x = plot_rect.right() - 170
            legend_y = plot_rect.top() + 8
            for idx, style in enumerate(series_styles):
                y = legend_y + idx * 20
                painter.setPen(QtGui.QPen(QtGui.QColor(style["color"]), 2))
                painter.drawLine(QtCore.QPointF(legend_x, y), QtCore.QPointF(legend_x + 26, y))
                painter.setPen(QtGui.QColor("#111827"))
                painter.drawText(QtCore.QRectF(legend_x + 32, y - 8, 132, 16), int(QtCore.Qt.AlignmentFlag.AlignLeft), style["label"])

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.fillRect(self.rect(), QtGui.QColor("#f8fafc"))
            if self.raw is None or self.recon is None:
                painter.setPen(QtGui.QColor("#111827"))
                painter.drawText(self.rect(), int(QtCore.Qt.AlignmentFlag.AlignCenter), "No linked profile arrays loaded.")
                painter.end()
                return

            raw_area, recon_area, x_profile_rect, y_profile_rect = self._layout_rects()
            profiles = linked_line_profiles(self.raw, self.recon, self.selected_x, self.selected_y)
            point = profiles["point"]
            self.selected_x = int(point["x_px"])
            self.selected_y = int(point["y_px"])

            self._draw_image_panel(painter, raw_area, self.raw_rect, self.raw, "Raw mean", self.raw.shape)
            self._draw_image_panel(painter, recon_area, self.recon_rect, self.recon, "Reconstruction", self.recon.shape)
            self._draw_profile_panel(painter, x_profile_rect, f"X profile at y={self.selected_y}", profiles["raw_x"], profiles["recon_x"])
            self._draw_profile_panel(painter, y_profile_rect, f"Y profile at x={self.selected_x}", profiles["raw_y"], profiles["recon_y"])

            legend = QtCore.QRectF(24, self.height() - 22, self.width() - 48, 18)
            painter.setPen(QtGui.QColor("#4b5563"))
            painter.drawText(legend, int(QtCore.Qt.AlignmentFlag.AlignRight), f"selected pixel: x={self.selected_x}, y={self.selected_y}")
            painter.end()

        def mousePressEvent(self, event):
            if self.raw is None or self.recon is None:
                return
            pos = event.position() if hasattr(event, "position") else event.pos()
            if self.raw_rect.contains(pos):
                rect = self.raw_rect
                shape = self.raw.shape
            elif self.recon_rect.contains(pos):
                rect = self.recon_rect
                shape = self.raw.shape
            else:
                return
            img_h, img_w = [max(1, int(v)) for v in shape[:2]]
            rel_x = (pos.x() - rect.left()) / max(1.0, rect.width())
            rel_y = (pos.y() - rect.top()) / max(1.0, rect.height())
            self.selected_x = max(0, min(img_w - 1, int(round(rel_x * (img_w - 1)))))
            self.selected_y = max(0, min(img_h - 1, int(round(rel_y * (img_h - 1)))))
            self.update()

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(APP_NAME)
            self._add_studio_navigation()
            self.current_frame = None
            self.preview_zoom = 1.0
            self.current_preview_kind = ""
            self.current_preview_path = None
            self.current_preview_title = ""
            self.current_preview_frame = None
            self.current_preview_draw_crop = True
            root = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(root)

            self.steps = QtWidgets.QListWidget()
            self.workflow_rows = [
                {"page": 0, "section": None},
                {"page": 1, "section": None},
                {"page": 2, "section": None},
                {"page": 3, "section": None},
                {"page": 4, "section": None},
                {"page": 5, "section": "main"},
                {"page": 5, "section": "main"},
                {"page": 5, "section": "diagnostics"},
                {"page": 5, "section": "progress"},
                {"page": 5, "section": "teaching"},
                {"page": 5, "section": "data"},
            ]
            self.steps.addItems(
                [
                    "1 Load Video",
                    "2 Calibration",
                    "3 Preprocess",
                    "4 Blinker Mode",
                    "5 Processing",
                    "6 Results",
                    "   Main Results",
                    "   Diagnostics",
                    "   Learning Progress",
                    "   Teaching Pipeline",
                    "   Data Files",
                ]
            )
            for row in range(6, self.steps.count()):
                item = self.steps.item(row)
                item.setForeground(QtGui.QColor("#4b5563"))
            self.steps.setFixedWidth(170)
            layout.addWidget(self.steps)

            center = QtWidgets.QWidget()
            center_layout = QtWidgets.QVBoxLayout(center)
            self.preview_title = QtWidgets.QLabel("Video Preview")
            self.preview_title.setStyleSheet("font-size: 18px; font-weight: 600;")
            self.result_tabs = QtWidgets.QTabBar()
            self.result_tabs.setExpanding(False)
            self.result_tabs.setVisible(False)
            self.result_tabs.currentChanged.connect(self.result_tab_changed)
            zoom_bar = QtWidgets.QWidget()
            zoom_layout = QtWidgets.QHBoxLayout(zoom_bar)
            zoom_layout.setContentsMargins(0, 0, 0, 0)
            zoom_layout.addStretch(1)
            self.zoom_out_button = QtWidgets.QPushButton("-")
            self.zoom_out_button.setToolTip("Zoom out")
            self.zoom_fit_button = QtWidgets.QPushButton("Fit")
            self.zoom_fit_button.setToolTip("Fit preview to the available viewer area")
            self.zoom_in_button = QtWidgets.QPushButton("+")
            self.zoom_in_button.setToolTip("Zoom in")
            self.zoom_label = QtWidgets.QLabel(preview_zoom_label(self.preview_zoom))
            self.zoom_label.setMinimumWidth(48)
            self.zoom_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            for button in (self.zoom_out_button, self.zoom_fit_button, self.zoom_in_button):
                button.setMaximumWidth(54)
                zoom_layout.addWidget(button)
            zoom_layout.addWidget(self.zoom_label)
            self.zoom_out_button.clicked.connect(self.zoom_out)
            self.zoom_fit_button.clicked.connect(self.zoom_fit)
            self.zoom_in_button.clicked.connect(self.zoom_in)
            self.preview = QtWidgets.QLabel("Browse or inspect a video to show the first frame here.")
            self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.preview.setMinimumSize(620, 360)
            self.preview.setStyleSheet("background: #111; color: #ddd; border: 1px solid #888;")
            self.preview_scroll = QtWidgets.QScrollArea()
            self.preview_scroll.setWidget(self.preview)
            self.preview_scroll.setWidgetResizable(False)
            self.preview_scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.document_preview = QtWidgets.QPlainTextEdit()
            self.document_preview.setReadOnly(True)
            self.document_preview.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            self.document_preview.setStyleSheet("background: #111; color: #eee; border: 1px solid #888; font-family: Consolas, monospace;")
            self.linked_profile_view = LinkedProfileWidget()
            self.viewer_stack = QtWidgets.QStackedWidget()
            self.viewer_stack.addWidget(self.preview_scroll)
            self.viewer_stack.addWidget(self.document_preview)
            self.viewer_stack.addWidget(self.linked_profile_view)
            self.log = QtWidgets.QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(150)
            self.log.setText("Ready.\nLoad a blinking video, inspect it, then run reconstruction.")
            center_layout.addWidget(self.preview_title)
            center_layout.addWidget(self.result_tabs)
            center_layout.addWidget(zoom_bar)
            center_layout.addWidget(self.viewer_stack, 1)
            center_layout.addWidget(self.log)
            layout.addWidget(center, 1)

            self.video = QtWidgets.QLineEdit()
            self.session_timestamp = current_timestamp()
            self.output_manually_selected = False
            self.output_label = QtWidgets.QLineEdit("")
            self.output_label.setPlaceholderText("optional, e.g. ABBE_loop")
            self.output_label.textChanged.connect(self.update_suggested_output_folder)
            self.out = QtWidgets.QLineEdit(str(reconstruction_default_output_dir("", self.session_timestamp)))
            browse = QtWidgets.QPushButton("Browse Video")
            browse.clicked.connect(self.browse_video)
            browse_output = QtWidgets.QPushButton("Browse Output Folder")
            browse_output.clicked.connect(self.browse_output_folder)
            inspect = QtWidgets.QPushButton("Inspect")
            inspect.clicked.connect(self.inspect_video)
            self.max_frames = QtWidgets.QSpinBox()
            self.max_frames.setRange(0, 100000)
            self.max_frames.setValue(0)
            self.mode = QtWidgets.QComboBox()
            self.mode.addItems(["auto", "single", "multiple"])
            self.processing_backend = QtWidgets.QComboBox()
            for backend in processing_backend_options():
                self.processing_backend.addItem(processing_backend_label(backend), backend)
            self.worker_count = QtWidgets.QSpinBox()
            self.worker_count.setRange(0, 256)
            self.worker_count.setValue(0)
            self.worker_count.setToolTip("CPU workers for reconstruction. 0 = auto, up to 90% of available CPUs.")
            self.processing_dependency_note = QtWidgets.QLabel("")
            self.processing_dependency_note.setWordWrap(True)
            self.processing_dependency_note.setStyleSheet("color: #374151; padding-top: 8px;")
            self.cumulative = QtWidgets.QLineEdit("")
            self.cumulative.setPlaceholderText("auto 10%,30%,60%,100% or e.g. 50,150,300,665")
            self.crop_enable = QtWidgets.QCheckBox("Enable crop")
            self.crop_x = QtWidgets.QSpinBox()
            self.crop_y = QtWidgets.QSpinBox()
            self.crop_w = QtWidgets.QSpinBox()
            self.crop_h = QtWidgets.QSpinBox()
            for spin in (self.crop_x, self.crop_y, self.crop_w, self.crop_h):
                spin.setRange(0, 100000)
            self.crop_w.setValue(0)
            self.crop_h.setValue(0)
            self.crop_enable.toggled.connect(self.update_crop_preview)
            for spin in (self.crop_x, self.crop_y, self.crop_w, self.crop_h):
                spin.valueChanged.connect(self.update_crop_preview)
            self.compare = QtWidgets.QCheckBox("Compare with truth after run")
            run = QtWidgets.QPushButton("Run Reconstruction")
            self.run_button = run
            self.reconstruction_worker = None
            run.clicked.connect(self.run_reconstruction_clicked)
            self.status = QtWidgets.QLabel("Ready")
            self.active_result_section = "main"
            self.current_run_results_ready = False
            self.allow_previous_results = QtWidgets.QCheckBox("Allow loading previous results")
            self.allow_previous_results.setChecked(False)
            self.allow_previous_results.toggled.connect(self.previous_results_permission_changed)
            self.result_sections = result_catalog(self.out.text(), include_existing=False)
            self.result_tab_items = []
            self.updating_result_tabs = False

            self.info = QtWidgets.QTextEdit()
            self.info.setReadOnly(True)
            self.info.setMinimumHeight(130)
            self.info.setText("Video info will appear here after Inspect.")

            # Raw-video playback for the 1 Load Video step.
            self.raw_video_frames = []
            self.raw_video_index = 0
            self.raw_video_loaded_path = ""
            self.raw_video_timer = QtCore.QTimer(self)
            self.raw_video_timer.timeout.connect(self.advance_raw_video_frame)
            self.raw_video_play_button = QtWidgets.QPushButton("Play video")
            self.raw_video_play_button.setEnabled(False)
            self.raw_video_play_button.setToolTip("Play the loaded blinking video frame by frame.")
            self.raw_video_play_button.clicked.connect(self.toggle_raw_video_playback)
            self.raw_video_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.raw_video_slider.setEnabled(False)
            self.raw_video_slider.setToolTip("Scrub through the loaded video.")
            self.raw_video_slider.valueChanged.connect(self.raw_video_slider_moved)
            self.raw_video_frame_label = QtWidgets.QLabel("No video loaded")
            self.raw_video_frame_label.setStyleSheet("color: #6b7280;")

            self.pages = QtWidgets.QStackedWidget()
            self.pages.addWidget(self._load_page(browse, browse_output, inspect))
            self.pages.addWidget(self._calibration_page())
            self.pages.addWidget(self._preprocess_page())
            self.pages.addWidget(self._blinker_page())
            self.pages.addWidget(self._processing_page())
            self.pages.addWidget(self._results_page(run))
            self.steps.currentRowChanged.connect(self.workflow_changed)
            self.steps.setCurrentRow(0)

            panel_box = QtWidgets.QVBoxLayout()
            title = QtWidgets.QLabel("Workflow Controls")
            title.setStyleSheet("font-size: 18px; font-weight: 600;")
            panel_box.addWidget(title)
            panel_box.addWidget(self.pages, 1)
            panel_box.addWidget(QtWidgets.QLabel("Video Info / Status"))
            panel_box.addWidget(self.info)
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
            self.processing_backend.currentIndexChanged.connect(self.update_processing_dependency_note)
            self.update_processing_dependency_note()

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
            if self.reconstruction_worker is not None and self.reconstruction_worker.isRunning():
                self.status.setText("Wait for reconstruction to finish before returning to Studio.")
                return
            script = studio_launcher_script()
            if not script.exists():
                QtWidgets.QMessageBox.warning(self, APP_NAME, f"Studio launcher not found:\n{script}")
                return
            subprocess.Popen([sys.executable, str(script)], cwd=str(PROJECT_ROOT))
            QtWidgets.QApplication.instance().quit()

        def _load_page(self, browse, browse_output, inspect):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            form.addRow("Video", self.video)
            form.addRow(browse)
            form.addRow("Output label", self.output_label)
            output_row = QtWidgets.QWidget()
            output_layout = QtWidgets.QHBoxLayout(output_row)
            output_layout.setContentsMargins(0, 0, 0, 0)
            output_layout.addWidget(self.out, 1)
            output_layout.addWidget(browse_output)
            form.addRow("Output", output_row)
            form.addRow(inspect)
            playback_row = QtWidgets.QWidget()
            playback_layout = QtWidgets.QHBoxLayout(playback_row)
            playback_layout.setContentsMargins(0, 0, 0, 0)
            playback_layout.addWidget(self.raw_video_play_button)
            playback_layout.addWidget(self.raw_video_slider, 1)
            form.addRow("Play video", playback_row)
            form.addRow(self.raw_video_frame_label)
            return page

        def _calibration_page(self):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            self.calibration_mode = QtWidgets.QComboBox()
            self.calibration_mode.addItems(["Pixels only", "Manual calibration"])
            self.calibration_unit = QtWidgets.QLineEdit("mm")
            self.calibration_scale = QtWidgets.QDoubleSpinBox()
            self.calibration_scale.setDecimals(9)
            self.calibration_scale.setRange(1e-9, 1e12)
            self.calibration_scale.setValue(1.0)
            self.calibration_scale.setEnabled(False)
            self.calibration_unit.setEnabled(False)
            self.calibration_mode.currentIndexChanged.connect(lambda i: self.calibration_scale.setEnabled(i == 1))
            self.calibration_mode.currentIndexChanged.connect(lambda i: self.calibration_unit.setEnabled(i == 1))
            form.addRow("Mode", self.calibration_mode)
            form.addRow("Unit label", self.calibration_unit)
            form.addRow("Unit per pixel", self.calibration_scale)
            note = QtWidgets.QLabel("Manual calibration adds x_calibrated and y_calibrated columns to the CSV, measured from the original frame origin. Images and line profiles retain pixel axes. Calibration does not estimate localization uncertainty.")
            note.setWordWrap(True)
            form.addRow(note)
            return page

        def _preprocess_page(self):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            self.background_correction = QtWidgets.QCheckBox("Subtract frame median background")
            self.background_correction.setChecked(False)
            self.threshold_quantile = QtWidgets.QDoubleSpinBox()
            self.threshold_quantile.setDecimals(4)
            self.threshold_quantile.setRange(0.0001, 1.0)
            self.threshold_quantile.setSingleStep(0.001)
            self.threshold_quantile.setValue(0.995)
            form.addRow(self.crop_enable)
            form.addRow("Crop x", self.crop_x)
            form.addRow("Crop y", self.crop_y)
            form.addRow("Crop width", self.crop_w)
            form.addRow("Crop height", self.crop_h)
            overlay = QtWidgets.QPushButton("Show crop overlay")
            overlay.clicked.connect(self.update_crop_preview)
            preview_crop = QtWidgets.QPushButton("Preview cropped region")
            preview_crop.clicked.connect(self.preview_cropped_region)
            crop_buttons = QtWidgets.QWidget()
            crop_buttons_layout = QtWidgets.QHBoxLayout(crop_buttons)
            crop_buttons_layout.setContentsMargins(0, 0, 0, 0)
            crop_buttons_layout.addWidget(overlay)
            crop_buttons_layout.addWidget(preview_crop)
            form.addRow(crop_buttons)
            form.addRow("Threshold quantile", self.threshold_quantile)
            form.addRow(self.background_correction)
            note = QtWidgets.QLabel("Blank or constant frames produce no localizations. Optional background subtraction uses the median of each cropped frame and clips negative signal to zero; this assumes sparse bright emitters. The raw mean remains uncorrected.")
            note.setWordWrap(True)
            form.addRow(note)
            return page

        def _blinker_page(self):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            form.addRow("Blinker mode", self.mode)
            note = QtWidgets.QLabel("Auto scans the movie and chooses single or multiple component localization. Multiple mode localizes separated bright components.")
            note.setWordWrap(True)
            form.addRow(note)
            return page

        def _processing_page(self):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            form.addRow("Processing backend", self.processing_backend)
            form.addRow("Workers (0=auto)", self.worker_count)
            form.addRow("Max frames (0=all)", self.max_frames)
            note = QtWidgets.QLabel("Serial is simplest. Auto CPU uses CPU parallel when useful, capped at about 90% of available CPUs. Only GPU uses the GPU localization path when supported. Auto CPU+GPU chooses GPU for single blinkers and CPU parallel for multiple blinkers. Auto CPU+GPU acceleration experimentally uses GPU thresholding plus CPU component localization for multiple blinkers.")
            note.setWordWrap(True)
            form.addRow(note)
            form.addRow("Dependencies", self.processing_dependency_note)
            return page

        def _results_page(self, run):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            form.addRow("Cumulative frames", self.cumulative)
            form.addRow(self.compare)
            self.truth_radius = QtWidgets.QDoubleSpinBox()
            self.truth_radius.setRange(0.001, 10000)
            self.truth_radius.setDecimals(3)
            self.truth_radius.setValue(2.0)
            form.addRow("Truth match radius (px)", self.truth_radius)
            form.addRow(self.allow_previous_results)
            load_results = QtWidgets.QPushButton("Load Previous Result Folder")
            load_results.clicked.connect(self.browse_result_folder)
            refresh_results = QtWidgets.QPushButton("Refresh Result Browser")
            refresh_results.clicked.connect(lambda: self.refresh_result_browser(show_first=True))
            self.load_previous_results_button = load_results
            self.load_previous_results_button.setEnabled(False)
            result_buttons = QtWidgets.QWidget()
            result_buttons_layout = QtWidgets.QHBoxLayout(result_buttons)
            result_buttons_layout.setContentsMargins(0, 0, 0, 0)
            result_buttons_layout.addWidget(load_results)
            result_buttons_layout.addWidget(refresh_results)
            form.addRow(run)
            form.addRow(result_buttons)
            return page

        @QtCore.Slot(int)
        def workflow_changed(self, row):
            if row < 0 or row >= len(self.workflow_rows):
                return
            if row != 0:
                self.stop_raw_video_playback()
            target = self.workflow_rows[row]
            self.pages.setCurrentIndex(int(target["page"]))
            section = target["section"]
            if section is None:
                self.result_tabs.setVisible(False)
                return
            self.active_result_section = str(section)
            self.refresh_result_browser(show_first=True)

        def _preview_viewport_size(self):
            size = self.preview_scroll.viewport().size()
            if size.width() <= 1 or size.height() <= 1:
                size = self.preview.size()
            return QtCore.QSize(max(1, int(size.width())), max(1, int(size.height())))

        def _scaled_size_for_zoom(self, source_size):
            viewport = self._preview_viewport_size()
            fitted = source_size.scaled(
                viewport,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            )
            return QtCore.QSize(
                max(1, int(round(fitted.width() * self.preview_zoom))),
                max(1, int(round(fitted.height() * self.preview_zoom))),
            )

        def _set_preview_pixmap(self, pix):
            self.viewer_stack.setCurrentWidget(self.preview_scroll)
            for button in (self.zoom_out_button, self.zoom_fit_button, self.zoom_in_button):
                button.setEnabled(True)
            self.zoom_label.setEnabled(True)
            self.preview.clear()
            self.preview.setText("")
            self.preview.setPixmap(pix)
            self.preview.setFixedSize(pix.size())
            self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        def _set_zoom(self, value):
            self.preview_zoom = clamp_preview_zoom(value)
            self.zoom_label.setText(preview_zoom_label(self.preview_zoom))
            self.render_current_preview()

        @QtCore.Slot()
        def zoom_out(self):
            self._set_zoom(self.preview_zoom / 1.25)

        @QtCore.Slot()
        def zoom_fit(self):
            self._set_zoom(1.0)

        @QtCore.Slot()
        def zoom_in(self):
            self._set_zoom(self.preview_zoom * 1.25)

        def remember_frame_preview(self, frame, title, draw_crop=True):
            self.current_preview_kind = "frame"
            self.current_preview_path = None
            self.current_preview_title = title
            self.current_preview_frame = frame
            self.current_preview_draw_crop = draw_crop

        def remember_file_preview(self, path, title, kind):
            self.current_preview_kind = str(kind)
            self.current_preview_path = Path(path)
            self.current_preview_title = title
            self.current_preview_frame = None
            self.current_preview_draw_crop = True

        def render_current_preview(self):
            if self.current_preview_kind == "frame" and self.current_preview_frame is not None:
                self._show_frame(self.current_preview_frame, self.current_preview_title, self.current_preview_draw_crop, remember=False)
            elif self.current_preview_kind == "image" and self.current_preview_path is not None:
                self._show_image_file(self.current_preview_path, self.current_preview_title, remember=False)
            elif self.current_preview_kind == "svg" and self.current_preview_path is not None:
                self._show_svg_file(self.current_preview_path, self.current_preview_title, remember=False)

        def _show_frame(self, frame, title, draw_crop=True, remember=True):
            if remember:
                self.remember_frame_preview(frame, title, draw_crop)
            arr = normalize_to_uint8(frame)
            h, w = arr.shape
            qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8).copy()
            pix = QtGui.QPixmap.fromImage(qimg)
            crop = self.preview_crop() if draw_crop else None
            if crop is not None:
                x, y, width, height = crop
                painter = QtGui.QPainter(pix)
                pen = QtGui.QPen(QtGui.QColor("#22d3ee"))
                pen.setWidth(max(2, int(round(max(w, h) / 220))))
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawRect(x, y, max(1, width - 1), max(1, height - 1))
                painter.end()
            scaled = pix.scaled(
                self._scaled_size_for_zoom(pix.size()),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self._set_preview_pixmap(scaled)
            self.preview_title.setText(title)

        def _show_image_file(self, path, title, remember=True):
            if remember:
                self.remember_file_preview(path, title, "image")
            pix = QtGui.QPixmap(str(path))
            if pix.isNull():
                self.log.append(f"Could not preview image: {path}")
                return
            scaled = pix.scaled(
                self._scaled_size_for_zoom(pix.size()),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self._set_preview_pixmap(scaled)
            self.preview_title.setText(title)

        def _show_svg_file(self, path, title, remember=True):
            if remember:
                self.remember_file_preview(path, title, "svg")
            try:
                from PySide6 import QtSvg
            except Exception as exc:
                self.log.append(f"SVG preview requires QtSvg: {exc}")
                self._show_text_file(path, title)
                return
            renderer = QtSvg.QSvgRenderer(str(path))
            if not renderer.isValid():
                self.log.append(f"Could not preview SVG: {path}")
                self._show_text_file(path, title)
                return
            default_size = renderer.defaultSize()
            if default_size.width() <= 0 or default_size.height() <= 0:
                default_size = QtCore.QSize(1000, 700)
            display_size = self._scaled_size_for_zoom(default_size)
            render_scale = 2
            image = QtGui.QImage(
                max(1, display_size.width() * render_scale),
                max(1, display_size.height() * render_scale),
                QtGui.QImage.Format.Format_ARGB32,
            )
            image.fill(QtGui.QColor("white"))
            painter = QtGui.QPainter(image)
            renderer.render(painter)
            painter.end()
            pix = QtGui.QPixmap.fromImage(image)
            scaled = pix.scaled(
                display_size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self._set_preview_pixmap(scaled)
            self.preview.setStyleSheet("background: white; color: #111; border: 1px solid #888;")
            self.preview_title.setText(title)

        def _show_text_file(self, path, title):
            self.remember_file_preview(path, title, "text")
            try:
                text = document_preview_text(path)
            except Exception as exc:
                self.log.append(f"Could not preview text file: {exc}")
                return
            self.viewer_stack.setCurrentWidget(self.document_preview)
            for button in (self.zoom_out_button, self.zoom_fit_button, self.zoom_in_button):
                button.setEnabled(False)
            self.zoom_label.setEnabled(False)
            self.document_preview.setPlainText(text)
            self.document_preview.moveCursor(QtGui.QTextCursor.MoveOperation.Start)
            self.document_preview.verticalScrollBar().setValue(0)
            self.document_preview.horizontalScrollBar().setValue(0)
            self.preview_title.setText(title)

        def _show_linked_profile_view(self, raw_path, title):
            raw_path = Path(raw_path)
            recon_path = raw_path.parent / "superres.npy"
            try:
                raw = np.load(raw_path)
                recon = np.load(recon_path)
                self.linked_profile_view.set_images(raw, recon)
            except Exception as exc:
                self.log.append(f"Could not open linked profile arrays: {exc}")
                return
            self.viewer_stack.setCurrentWidget(self.linked_profile_view)
            for button in (self.zoom_out_button, self.zoom_fit_button, self.zoom_in_button):
                button.setEnabled(False)
            self.zoom_label.setEnabled(False)
            self.preview_title.setText(title)

        def _reset_image_preview_style(self):
            self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.preview.setStyleSheet("background: #111; color: #ddd; border: 1px solid #888;")

        @QtCore.Slot()
        def browse_video(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose blinking video", "", "Videos (*.mp4 *.avi *.mov);;All files (*.*)")
            if path:
                self.video.setText(path)
                self.inspect_video()

        @QtCore.Slot()
        def browse_output_folder(self):
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder for this reconstruction", self.out.text())
            if folder:
                self.output_manually_selected = True
                self.out.setText(folder)
                self.current_run_results_ready = False
                self.refresh_result_browser(show_first=False)
                self.status.setText("Output folder selected. Results will appear after reconstruction.")

        @QtCore.Slot()
        def browse_result_folder(self):
            if not self.allow_previous_results.isChecked():
                self.status.setText("Check 'Allow loading previous results' before opening old result folders.")
                return
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose reconstruction output folder", self.out.text())
            if folder:
                self.output_manually_selected = True
                self.out.setText(folder)
                self.current_run_results_ready = False
                self.refresh_result_browser(show_first=True)

        @QtCore.Slot(bool)
        def previous_results_permission_changed(self, checked):
            self.load_previous_results_button.setEnabled(bool(checked))
            if not checked and not self.current_run_results_ready:
                self.refresh_result_browser(show_first=False)
                self.status.setText("Previous results hidden.")

        @QtCore.Slot()
        def inspect_video(self):
            self._reset_raw_video_playback()
            try:
                info = read_video_info(self.video.text())
                frames = read_video_frames(self.video.text(), max_frames=1)
                if frames:
                    self.current_frame = frames[0]
                    if self.crop_w.value() == 0:
                        self.crop_w.setValue(int(info["width"]))
                    if self.crop_h.value() == 0:
                        self.crop_h.setValue(int(info["height"]))
                    self._reset_image_preview_style()
                    self._show_frame(self.current_frame, "First Frame Preview")
                    self.raw_video_play_button.setEnabled(True)
                    self.raw_video_frame_label.setText("Press the Play video button to watch all frames.")
                text = (
                    f"Video: {self.video.text()}\n"
                    f"Frames: {info['frames']}\n"
                    f"Size: {info['width']} x {info['height']}\n"
                    f"FPS: {info['fps']:.3f}"
                )
                self.info.setText(text)
                self.log.append(text)
                self.status.setText("Video inspected")
            except Exception as exc:
                self.status.setText(str(exc))
                self.log.append(f"Inspect error: {exc}")

        def _reset_raw_video_playback(self):
            self.stop_raw_video_playback()
            self.raw_video_frames = []
            self.raw_video_index = 0
            self.raw_video_loaded_path = ""
            self.raw_video_play_button.setEnabled(False)
            self.raw_video_slider.blockSignals(True)
            self.raw_video_slider.setValue(0)
            self.raw_video_slider.setEnabled(False)
            self.raw_video_slider.blockSignals(False)
            self.raw_video_frame_label.setText("No video loaded")

        def _ensure_raw_video_frames(self):
            path = self.video.text().strip()
            if not path:
                self.status.setText("Choose a video first.")
                return False
            if self.raw_video_frames and self.raw_video_loaded_path == path:
                return True
            self.status.setText("Loading video frames for playback...")
            QtWidgets.QApplication.processEvents()
            try:
                frames = read_video_frames(path, max_frames=RAW_VIDEO_PLAYBACK_CAP)
            except Exception as exc:
                self.status.setText(f"Could not load video for playback: {exc}")
                self.log.append(f"Playback load error: {exc}")
                return False
            if not frames:
                self.status.setText("No frames were found in this video.")
                return False
            self.raw_video_frames = frames
            self.raw_video_loaded_path = path
            self.raw_video_index = 0
            self.raw_video_slider.blockSignals(True)
            self.raw_video_slider.setRange(0, len(frames) - 1)
            self.raw_video_slider.setValue(0)
            self.raw_video_slider.setEnabled(True)
            self.raw_video_slider.blockSignals(False)
            try:
                total = int(read_video_info(path).get("frames", 0))
            except Exception:
                total = 0
            if total > len(frames):
                self.log.append(f"Playback shows the first {len(frames)} of {total} frames.")
            self.status.setText(f"Loaded {len(frames)} frames for playback.")
            self._reset_image_preview_style()
            self.show_raw_video_frame()
            return True

        @QtCore.Slot()
        def toggle_raw_video_playback(self):
            if self.raw_video_timer.isActive():
                self.raw_video_timer.stop()
                self.raw_video_play_button.setText("Play video")
                return
            if not self._ensure_raw_video_frames():
                return
            self.raw_video_timer.start(120)
            self.raw_video_play_button.setText("Pause")

        @QtCore.Slot()
        def advance_raw_video_frame(self):
            if not self.raw_video_frames:
                self.stop_raw_video_playback()
                return
            self.raw_video_index = (self.raw_video_index + 1) % len(self.raw_video_frames)
            self.show_raw_video_frame()

        @QtCore.Slot(int)
        def raw_video_slider_moved(self, value):
            if not self.raw_video_frames:
                return
            self.raw_video_index = max(0, min(int(value), len(self.raw_video_frames) - 1))
            self.show_raw_video_frame()

        def show_raw_video_frame(self):
            if not self.raw_video_frames:
                return
            total = len(self.raw_video_frames)
            frame = self.raw_video_frames[self.raw_video_index]
            self._show_frame(frame, "Loaded Video", draw_crop=False)
            self.raw_video_frame_label.setText(f"Frame {self.raw_video_index + 1} / {total}")
            if self.raw_video_slider.value() != self.raw_video_index:
                self.raw_video_slider.blockSignals(True)
                self.raw_video_slider.setValue(self.raw_video_index)
                self.raw_video_slider.blockSignals(False)

        def stop_raw_video_playback(self):
            if self.raw_video_timer.isActive():
                self.raw_video_timer.stop()
            if hasattr(self, "raw_video_play_button"):
                self.raw_video_play_button.setText("Play video")

        @QtCore.Slot()
        def update_crop_preview(self):
            if self.current_frame is None:
                return
            title = "First Frame Preview with Crop" if self.preview_crop() is not None else "First Frame Preview"
            self._show_frame(self.current_frame, title)

        @QtCore.Slot()
        def preview_cropped_region(self):
            if self.current_frame is None:
                self.status.setText("Inspect a video before previewing crop.")
                return
            crop = self.preview_crop()
            if crop is None:
                self.status.setText("Crop is disabled or empty.")
                return
            x, y, width, height = crop
            cropped = self.current_frame[y:y + height, x:x + width]
            self._show_frame(cropped, "Cropped Region Preview", draw_crop=False)
            self.status.setText(f"Previewing crop x={x}, y={y}, width={width}, height={height}")

        @QtCore.Slot()
        def run_reconstruction_clicked(self):
            if self.reconstruction_worker is not None and self.reconstruction_worker.isRunning():
                return
            try:
                max_frames = int(self.max_frames.value()) or None
                kwargs = dict(
                    video_path=self.video.text(),
                    output_dir=self.out.text(),
                    max_frames=max_frames,
                    blinker_mode=self.mode.currentText(),
                    compare_truth=self.compare.isChecked(),
                    cumulative_frames=parse_cumulative_frames(self.cumulative.text()),
                    crop=self.current_crop(),
                    processing_backend=self.current_processing_backend(),
                    worker_count=int(self.worker_count.value()) or None,
                    threshold_quantile=self.threshold_quantile.value(),
                    pixel_size=self.calibration_scale.value() if self.calibration_mode.currentIndex() == 1 else None,
                    unit=self.calibration_unit.text(),
                    background_mode="frame_median" if self.background_correction.isChecked() else "none",
                    truth_radius_px=self.truth_radius.value(),
                )
                self.pages.setEnabled(False)
                self.back_to_studio_button.setEnabled(False)
                self.raw_video_timer.stop()
                self.status.setText("Reconstructing and exporting figures...")
                self.log.append("Reconstruction started. The window remains responsive.")
                self.reconstruction_worker = ReconstructionWorker(kwargs, self)
                self.reconstruction_worker.completed.connect(self.reconstruction_completed)
                self.reconstruction_worker.failed.connect(self.reconstruction_failed)
                self.reconstruction_worker.finished.connect(self.reconstruction_finished)
                self.reconstruction_worker.start()
            except Exception as exc:
                self.reconstruction_failed(str(exc))
                self.reconstruction_finished()

        @QtCore.Slot(dict)
        def reconstruction_completed(self, summary):
            self.info.setText(str(summary))
            self.log.append(f"Saved outputs in {self.out.text()}")
            self.current_run_results_ready = True
            self.steps.setCurrentRow(5)
            self.active_result_section = "main"
            self.refresh_result_browser(show_first=True)
            self.status.setText(f"Saved outputs in {self.out.text()}")

        @QtCore.Slot(str)
        def reconstruction_failed(self, message):
            self.status.setText(message)
            self.log.append(f"Run error: {message}")

        @QtCore.Slot()
        def reconstruction_finished(self):
            self.pages.setEnabled(True)
            self.back_to_studio_button.setEnabled(True)
            if self.reconstruction_worker is not None:
                self.reconstruction_worker.deleteLater()
                self.reconstruction_worker = None

        def closeEvent(self, event):
            if self.reconstruction_worker is not None and self.reconstruction_worker.isRunning():
                self.status.setText("Reconstruction is running. Close the window after it finishes.")
                event.ignore()
            else:
                event.accept()

        @QtCore.Slot()
        def refresh_result_browser(self, show_first=True):
            include_existing = self.current_run_results_ready or self.allow_previous_results.isChecked()
            self.result_sections = result_catalog(self.out.text(), include_existing=include_existing)
            section = next((item for item in self.result_sections if item["key"] == self.active_result_section), None)
            self.updating_result_tabs = True
            while self.result_tabs.count():
                self.result_tabs.removeTab(0)
            self.result_tab_items = []
            if section is not None:
                self.preview_title.setText(section["label"])
                for item in section["items"]:
                    self.result_tabs.addTab(item["label"])
                    self.result_tab_items.append(item)
            self.result_tabs.setVisible(bool(self.result_tab_items))
            self.updating_result_tabs = False
            if not self.result_tab_items:
                self.viewer_stack.setCurrentWidget(self.preview_scroll)
                for button in (self.zoom_out_button, self.zoom_fit_button, self.zoom_in_button):
                    button.setEnabled(False)
                self.zoom_label.setEnabled(False)
                self.preview.clear()
                if include_existing:
                    message = f"No generated files found for {section['label'] if section else 'this section'}.\nRun reconstruction or choose a previous result folder."
                else:
                    message = "No results loaded yet.\nRun reconstruction to show new results, or check 'Allow loading previous results' and load an old folder."
                self.preview.setText(message)
                self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.status.setText("No current results loaded")
                return
            if show_first:
                self.result_tabs.setCurrentIndex(0)
                self.show_result_item(self.result_tab_items[0])

        @QtCore.Slot(int)
        def result_tab_changed(self, index):
            if self.updating_result_tabs or index < 0 or index >= len(self.result_tab_items):
                return
            self.show_result_item(self.result_tab_items[index])

        def show_result_item(self, item):
            path = item["path"]
            self.preview_zoom = 1.0
            self.zoom_label.setText(preview_zoom_label(self.preview_zoom))
            if item["kind"] == "image":
                self._reset_image_preview_style()
                self._show_image_file(path, item["label"])
            elif item["kind"] == "svg":
                self._show_svg_file(path, item["label"])
            elif item["kind"] == "linked_profile":
                self._show_linked_profile_view(path, item["label"])
            else:
                self._show_text_file(path, item["label"])
            self.status.setText(f"Showing {item['label']}")

        @QtCore.Slot()
        def update_suggested_output_folder(self, *_args):
            if self.output_manually_selected or self.current_run_results_ready:
                return
            self.out.setText(str(reconstruction_default_output_dir(self.output_label.text(), self.session_timestamp)))

        def current_crop(self):
            if not self.crop_enable.isChecked():
                return None
            width = int(self.crop_w.value())
            height = int(self.crop_h.value())
            if width <= 0 or height <= 0:
                raise ValueError("Crop is enabled, but width/height are zero.")
            crop = (int(self.crop_x.value()), int(self.crop_y.value()), width, height)
            if self.current_frame is None:
                return crop
            clipped = clip_crop_to_shape(crop, self.current_frame.shape)
            if clipped is None:
                raise ValueError("Crop region is outside the loaded frame.")
            return clipped

        def preview_crop(self):
            if self.current_frame is None or not self.crop_enable.isChecked():
                return None
            return clip_crop_to_shape(
                (int(self.crop_x.value()), int(self.crop_y.value()), int(self.crop_w.value()), int(self.crop_h.value())),
                self.current_frame.shape,
            )

        def current_processing_backend(self):
            return str(self.processing_backend.currentData() or "serial")

        @QtCore.Slot()
        def update_processing_dependency_note(self):
            self.processing_dependency_note.setText(gpu_dependency_guidance(self.current_processing_backend()))

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    return app.exec()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--run-video", help="Run reconstruction without opening the GUI.")
    parser.add_argument("--out", default="")
    parser.add_argument("--label", default="", help="Optional output folder label used when --out is not supplied.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument("--multi-emitter", action="store_true")
    parser.add_argument("--blinker-mode", choices=["auto", "single", "multiple"], default=None)
    parser.add_argument("--crop", default="", help="Optional crop as x,y,width,height.")
    parser.add_argument("--compare-truth", action="store_true")
    parser.add_argument("--cumulative-frames", default="", help="Optional four comma-separated cumulative frame values, e.g. 50,150,300,665.")
    parser.add_argument("--processing-backend", choices=processing_backend_options(), default="serial")
    parser.add_argument("--workers", type=int, default=0, help="Worker count for CPU parallel processing. Use 0 for auto.")
    parser.add_argument("--pixel-size", type=float, default=None, help="Physical units per camera pixel; calibrated CSV coordinates use the original frame origin.")
    parser.add_argument("--unit", default="px", help="Unit label for calibrated CSV coordinates, e.g. mm or nm.")
    parser.add_argument("--background-mode", choices=["none", "frame_median"], default="none")
    parser.add_argument("--truth-radius-px", type=float, default=2.0)
    return parser.parse_args(argv)


def parse_cumulative_frames(text: str) -> list[int] | None:
    text = (text or "").strip()
    if not text:
        return None
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(float(part)))
    return values or None


def parse_crop(text: str) -> tuple[int, int, int, int] | None:
    text = (text or "").strip()
    if not text:
        return None
    parts = [int(float(part.strip())) for part in text.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("--crop must contain four comma-separated values: x,y,width,height")
    return tuple(parts)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.run_video:
        output_dir = args.out or reconstruction_default_output_dir(args.label)
        summary = run_reconstruction(
            args.run_video,
            output_dir,
            max_frames=args.max_frames or None,
            threshold_quantile=args.threshold_quantile,
            blinker_mode=args.blinker_mode or ("multiple" if args.multi_emitter else "single"),
            compare_truth=args.compare_truth,
            cumulative_frames=parse_cumulative_frames(args.cumulative_frames),
            crop=parse_crop(args.crop),
            processing_backend=args.processing_backend,
            worker_count=args.workers or None,
            pixel_size=args.pixel_size,
            unit=args.unit,
            background_mode=args.background_mode,
            truth_radius_px=args.truth_radius_px,
        )
        print(f"Saved reconstruction outputs to {output_dir}")
        print(f"Frames processed: {summary['frames_processed']}")
        print(f"Localizations: {summary['localizations']}")
        if "truth_comparison" in summary:
            truth = summary["truth_comparison"]
            print(f"Truth comparison: precision={truth['precision']}, recall={truth['recall']}, F1={truth['f1']}")
        return 0
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
