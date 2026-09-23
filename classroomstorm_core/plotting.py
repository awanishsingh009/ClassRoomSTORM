from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np


def _try_matplotlib():
    if os.environ.get("CLASSROOMSTORM_DISABLE_MATPLOTLIB", "").lower() in {"1", "true", "yes"}:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from .figure_style import configure
        except ImportError:
            from figure_style import configure
        configure(plt)
        return plt
    except Exception:
        return None


def save_grayscale_png(path: str | Path, image: np.ndarray) -> Path:
    from PIL import Image
    try:
        from .simulation import normalize_to_uint8
    except ImportError:
        from simulation import normalize_to_uint8

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalize_to_uint8(image)).save(target)
    return target


def save_superres_png(path: str | Path, image: np.ndarray, title: str = "Super-Resolution") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt = _try_matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor="white")
        im = ax.imshow(image, cmap="viridis", interpolation="nearest")
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Kernel weight per camera pixel")
        fig.tight_layout()
        fig.savefig(target, dpi=300)
        plt.close(fig)
        return target
    return save_grayscale_png(target, image)


def save_side_by_side(path: str | Path, left: np.ndarray, right: np.ndarray, left_title: str, right_title: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt = _try_matplotlib()
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), facecolor="white")
        axes[0].imshow(left, cmap="gray", interpolation="nearest")
        axes[0].set_title(left_title, fontsize=13, pad=12)
        axes[0].axis("off")
        axes[1].imshow(right, cmap="gray", interpolation="nearest")
        axes[1].set_title(right_title, fontsize=13, pad=12)
        axes[1].axis("off")
        fig.text(.5, .02, "Same field of view; each image uses its own intensity scale. Render width is a display choice.", ha="center", fontsize=9)
        fig.tight_layout(w_pad=3.0, rect=(0,.06,1,1))
        fig.savefig(target, dpi=300)
        plt.close(fig)
        return target

    from PIL import Image, ImageDraw
    try:
        from .simulation import normalize_to_uint8
    except ImportError:
        from simulation import normalize_to_uint8

    left_img = Image.fromarray(normalize_to_uint8(left)).convert("RGB")
    right_img = Image.fromarray(normalize_to_uint8(right)).convert("RGB")
    if left_img.size != right_img.size:
        right_img = right_img.resize(left_img.size)
    title_h = 28
    gap = 12
    canvas = Image.new("RGB", (left_img.width * 2 + gap, left_img.height + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), left_title, fill="black")
    draw.text((left_img.width + gap + 4, 6), right_title, fill="black")
    canvas.paste(left_img, (0, title_h))
    canvas.paste(right_img, (left_img.width + gap, title_h))
    canvas.save(target)
    return target


def save_counts_per_frame(path: str | Path, localizations: Sequence[dict]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = [int(loc["frame"]) for loc in localizations]
    if frames:
        max_frame = max(frames)
        counts = np.bincount(frames, minlength=max_frame + 1)
    else:
        counts = np.array([], dtype=int)
    plt = _try_matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="white")
        ax.plot(np.arange(len(counts)), counts, color="#0072B2", linewidth=1.5)
        ax.set_title("Localizations per Frame")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Count")
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, max(1, int(counts.max()) if len(counts) else 1) * 1.15)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(target, dpi=300)
        plt.close(fig)
        return target
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (640, 320), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), "Localizations per Frame", fill="black")
    if len(counts) > 0 and counts.max() > 0:
        x0, y0, w, h = 40, 50, 560, 230
        draw.rectangle((x0, y0, x0 + w, y0 + h), outline="black")
        pts = []
        for i, c in enumerate(counts):
            x = x0 + int(i / max(1, len(counts) - 1) * w)
            y = y0 + h - int(c / max(1, int(counts.max())) * h)
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=(11, 114, 133), width=2)
    canvas.save(target)
    return target


def save_intensity_histogram(path: str | Path, localizations: Sequence[dict]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    vals = [float(loc.get("intensity", loc.get("N", 0.0))) for loc in localizations]
    plt = _try_matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
        ax.hist(vals, bins=min(40, max(5, int(np.sqrt(max(1, len(vals)))))), color="#0072B2", alpha=0.85)
        ax.set_title("Selected-pixel Intensity Histogram")
        ax.set_xlabel("Summed selected-pixel intensity (camera units)")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(target, dpi=300)
        plt.close(fig)
        return target
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (520, 360), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), "Selected-pixel Intensity Histogram", fill="black")
    if vals:
        hist, _ = np.histogram(vals, bins=min(30, max(5, int(np.sqrt(len(vals))))))
        x0, y0, w, h = 40, 60, 430, 240
        draw.rectangle((x0, y0, x0 + w, y0 + h), outline="black")
        bw = max(1, w // len(hist))
        for i, c in enumerate(hist):
            bar_h = int(c / max(1, int(hist.max())) * h)
            draw.rectangle((x0 + i * bw, y0 + h - bar_h, x0 + (i + 1) * bw - 1, y0 + h), fill=(54, 79, 199))
    canvas.save(target)
    return target


def save_density_map(path: str | Path, localizations: Sequence[dict], shape: tuple[int, int]) -> Path:
    try:
        from .reconstruction import render_localization_image
    except ImportError:
        from reconstruction import render_localization_image

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    density = render_localization_image(localizations, shape, sigma_px=3.0)
    plt = _try_matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor="white")
        im = ax.imshow(density, cmap="viridis", interpolation="nearest")
        ax.set_title("Rendered density (display sigma = 3 px)", fontsize=12)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Kernel weight per camera pixel")
        fig.tight_layout()
        fig.savefig(target, dpi=300)
        plt.close(fig)
        return target
    return save_grayscale_png(target, density)


def save_raw_localization_overlay(path: str | Path, raw_image: np.ndarray, localizations: Sequence[dict]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    xs = [float(loc["x_px"]) for loc in localizations]
    ys = [float(loc["y_px"]) for loc in localizations]
    plt = _try_matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor="white")
        ax.imshow(raw_image, cmap="gray", interpolation="nearest")
        ax.scatter(xs, ys, s=8, c="#D55E00", alpha=0.8, linewidths=0)
        ax.set_title("Raw Mean with Localizations", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(target, dpi=300)
        plt.close(fig)
        return target

    from PIL import Image, ImageDraw
    try:
        from .simulation import normalize_to_uint8
    except ImportError:
        from simulation import normalize_to_uint8

    canvas = Image.fromarray(normalize_to_uint8(raw_image)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for x, y in zip(xs, ys):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(34, 211, 238))
    canvas.save(target)
    return target


def cumulative_checkpoints(total_frames: int, manual: Sequence[int] | None = None) -> list[int]:
    """Return four cumulative frame cutoffs for the reconstruction-emergence plot."""
    total = max(1, int(total_frames))
    if manual:
        values = [max(1, min(total, int(v))) for v in manual if int(v) > 0]
        values = sorted(values)
        if not values:
            return cumulative_checkpoints(total)
        while len(values) < 4:
            values.append(total)
        return values[:4]
    return [
        max(1, int(round(total * 0.10))),
        max(1, int(round(total * 0.30))),
        max(1, int(round(total * 0.60))),
        total,
    ]


def cumulative_frame_panels(
    localizations: Sequence[dict],
    total_frames: int | None = None,
    manual: Sequence[int] | None = None,
) -> list[dict]:
    """Return frame-based cumulative localization panels for plotting."""
    locs = list(localizations)
    inferred_total = 0
    if locs:
        inferred_total = max(int(loc.get("frame", 0)) for loc in locs) + 1
    total = max(1, int(total_frames if total_frames is not None else inferred_total or 1))
    cuts = cumulative_checkpoints(total, manual=manual)
    panels = []
    for cutoff in cuts:
        selected = [loc for loc in locs if int(loc.get("frame", 0)) < int(cutoff)]
        label = "All frames" if int(cutoff) >= total else f"{int(cutoff)} frame" + ("" if int(cutoff) == 1 else "s")
        panels.append({"frame_cutoff": int(cutoff), "label": label, "localizations": selected})
    return panels


def save_cumulative_reconstruction_panel(
    path: str | Path,
    localizations: Sequence[dict],
    shape: tuple[int, int],
    manual_checkpoints: Sequence[int] | None = None,
    total_frames: int | None = None,
) -> Path:
    try:
        from .reconstruction import render_localization_image
    except ImportError:
        from reconstruction import render_localization_image

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    panels_data = cumulative_frame_panels(localizations, total_frames=total_frames, manual=manual_checkpoints)
    labels = [panel["label"] for panel in panels_data]
    images = [render_localization_image(panel["localizations"], shape, sigma_px=1.5) for panel in panels_data]
    plt = _try_matplotlib()
    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), facecolor="white")
        axes_flat = list(np.ravel(axes))
        vmax = max(float(np.max(img)) for img in images) if images else 1.0
        for ax, img, label in zip(axes_flat, images, labels):
            ax.imshow(img, cmap="viridis", vmin=0, vmax=max(vmax, 1e-9), interpolation="nearest")
            ax.set_title(label, fontsize=10)
            ax.axis("off")
        fig.suptitle("Cumulative reconstruction: shared scale, display sigma = 1.5 px", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(target, dpi=300)
        plt.close(fig)
        return target

    from PIL import Image, ImageDraw
    try:
        from .simulation import normalize_to_uint8
    except ImportError:
        from simulation import normalize_to_uint8

    vmax = max((float(np.max(img)) for img in images), default=1.0)
    panels = [Image.fromarray(np.clip(img / max(vmax, 1e-9) * 255, 0, 255).astype(np.uint8)).convert("RGB") for img in images]
    if not panels:
        panels = [Image.new("RGB", (shape[1], shape[0]), "black")]
        labels = ["No localizations"]
    title_h = 28
    gap = 12
    panel_w = max(p.width for p in panels)
    panel_h = max(p.height for p in panels)
    width = panel_w * 2 + gap
    height = (panel_h + title_h) * 2 + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (panel, label) in enumerate(zip(panels, labels)):
        col = idx % 2
        row = idx // 2
        x0 = col * (panel_w + gap)
        y0 = row * (panel_h + title_h + gap)
        draw.text((x0 + 4, y0 + 6), label, fill="black")
        canvas.paste(panel, (x0, y0 + title_h))
    canvas.save(target)
    return target


def save_line_profile(path: str | Path, raw_image: np.ndarray, reconstruction: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(raw_image, dtype=float)
    recon = np.asarray(reconstruction, dtype=float)
    y = raw.shape[0] // 2
    raw_profile = raw[y, :]
    recon_profile = recon[min(y, recon.shape[0] - 1), :]
    if raw_profile.max() > raw_profile.min():
        raw_profile = (raw_profile - raw_profile.min()) / (raw_profile.max() - raw_profile.min())
    else:
        raw_profile = np.zeros_like(raw_profile)
    if recon_profile.max() > recon_profile.min():
        recon_profile = (recon_profile - recon_profile.min()) / (recon_profile.max() - recon_profile.min())
    else:
        recon_profile = np.zeros_like(recon_profile)
    x = np.arange(len(raw_profile))
    plt = _try_matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 3.8), facecolor="white")
        ax.plot(x, raw_profile, label="Raw mean", color="#c1272d", linewidth=1.6, linestyle="--")
        ax.plot(x, recon_profile, label="Reconstruction", color="#0072B2", linewidth=1.6)
        ax.set_title(f"Camera row {y}: displayed-image comparison")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("Independently min-max normalized signal")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(target, dpi=300)
        plt.close(fig)
        return target

    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (640, 340), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), "Central Line Profile", fill="black")
    x0, y0, w, h = 42, 58, 560, 230
    draw.rectangle((x0, y0, x0 + w, y0 + h), outline="black")
    for profile, color in [(raw_profile, (73, 80, 87)), (recon_profile, (11, 114, 133))]:
        pts = []
        for i, val in enumerate(profile):
            px = x0 + int(i / max(1, len(profile) - 1) * w)
            py = y0 + h - int(float(val) * h)
            pts.append((px, py))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)
    canvas.save(target)
    return target


def save_localization_scatter(path: str | Path, localizations: Sequence[dict]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    xs = [float(loc["x_px"]) for loc in localizations]
    ys = [float(loc["y_px"]) for loc in localizations]
    plt = _try_matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xs, ys, s=8, alpha=0.8)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(target, dpi=300)
        plt.close(fig)
        return target

    from PIL import Image, ImageDraw

    size = 480
    margin = 32
    canvas = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((margin, margin, size - margin, size - margin), outline="black")
    if xs and ys:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span = max(1.0, x_max - x_min)
        y_span = max(1.0, y_max - y_min)
        for x, y in zip(xs, ys):
            px = margin + int((x - x_min) / x_span * (size - 2 * margin))
            py = margin + int((y - y_min) / y_span * (size - 2 * margin))
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(30, 105, 180))
    draw.text((margin, 8), "Localization scatter (px)", fill="black")
    canvas.save(target)
    return target
