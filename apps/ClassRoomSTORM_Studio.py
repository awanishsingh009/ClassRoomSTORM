"""
ClassRoomSTORM Studio V1.1.

Animated launch screen for choosing between the virtual experiment and
reconstruction workflows.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from PySide6 import QtCore, QtGui, QtWidgets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "ClassRoomSTORM Studio"
APP_VERSION = "V1.1"
AUTHOR_CREDIT = "Developed by Dr. Awanish Pratap Singh"
AUTHOR_AFFILIATION = "Institute of Biomedical Optics, University of Lübeck"
STUDIO_TRANSITION_MS = 650


def studio_workflows() -> list[dict]:
    return [
        {
            "key": "virtual",
            "title": "Virtual Experiment",
            "subtitle": "Create blinking data from a logo, letter, or pattern",
            "script": PROJECT_ROOT / "apps" / "ClassRoomSTORM_VirtualExperiment.py",
        },
        {
            "key": "reconstruction",
            "title": "SR Reconstruction",
            "subtitle": "Reconstruct from recorded blinking video or stack",
            "script": PROJECT_ROOT / "apps" / "ClassRoomSTORM_Reconstruction.py",
        },
    ]


def studio_reader_links() -> list[dict]:
    return [
        {
            "key": "help",
            "label": "Help",
            "path": PROJECT_ROOT / "docs" / "ClassRoomSTORM_Help.pdf",
        },
        {
            "key": "theory",
            "label": "Theory",
            "path": PROJECT_ROOT / "docs" / "ClassRoomSTORM_Theory.pdf",
        },
    ]


def reader_display_path(reader: dict) -> Path:
    path = Path(reader["path"])
    if path.exists():
        return path
    fallback = Path(reader.get("fallback_path", path))
    return fallback


def studio_footer_text() -> str:
    return f"{APP_VERSION} | {AUTHOR_CREDIT} | {AUTHOR_AFFILIATION}"


def equation_fragments() -> list[str]:
    return [
        r"$d \approx \frac{\lambda}{2\,\mathrm{NA}}$",
        r"$r \approx \frac{0.61\,\lambda}{\mathrm{NA}}$",
        r"$k_{\max}=\frac{2\,\mathrm{NA}}{\lambda}$",
        r"$h(x,y)=h_0\exp\!\left[-\frac{r^2}{2\sigma^2}\right]$",
        r"$g(x,y)=h(x,y)\ast f(x,y)+\eta(x,y)$",
        r"$\mu_i=B+A\exp\!\left[-\frac{(x_i-x_0)^2+(y_i-y_0)^2}{2s^2}\right]$",
        r"$y_i \sim \mathrm{Poisson}(\mu_i)$",
        r"$L(\theta)=\sum_i\!\left[\mu_i(\theta)-y_i\ln\mu_i(\theta)\right]$",
        r"$\mathrm{Var}(\hat{\theta}) \geq F^{-1}(\theta)$",
        r"$F_{mn}=\sum_i\frac{1}{\mu_i}\frac{\partial\mu_i}{\partial\theta_m}\frac{\partial\mu_i}{\partial\theta_n}$",
        r"$\sigma_{\mathrm{loc}}^2\approx\frac{s^2+a^2/12}{N}+\sigma_{\mathrm{bg}}^2$",
        r"$R_{\mathrm{Nyquist}}\approx\frac{2}{\sqrt{\rho}}$",
        r"$G(k_x,k_y)=H(k_x,k_y)F(k_x,k_y)$",
    ]


def studio_mathtext_rc() -> dict[str, object]:
    """Matplotlib settings matching the paper figures' Computer Modern math."""
    return {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
    }


def studio_equation_font_sizes() -> tuple[int, ...]:
    return (10, 12, 14, 16)


def studio_reference_style() -> dict[str, object]:
    return {
        "background_base_colors": ((0, 0, 0), (5, 6, 10), (0, 0, 0)),
        "equation_drift_px": (16.0, 9.0),
        "equation_speed_range": (0.018, 0.055),
        "equation_zoom_range": (0.86, 1.08),
        "equation_max_width_fraction": 0.34,
        "psf_halo_multiplier": 4.6,
        "background_bottom_right_haze": (18, 20, 32, 46),
    }


def studio_equation_layout_slots() -> tuple[tuple[float, float], ...]:
    return (
        (0.10, 0.06),
        (0.31, 0.06),
        (0.52, 0.06),
        (0.73, 0.06),
        (0.04, 0.18),
        (0.04, 0.38),
        (0.04, 0.58),
        (0.04, 0.78),
        (0.68, 0.18),
        (0.68, 0.38),
        (0.68, 0.58),
        (0.68, 0.78),
        (0.10, 0.91),
        (0.31, 0.91),
        (0.52, 0.91),
        (0.73, 0.91),
    )


def clamped_equation_rect(x: float, y: float, width: float, height: float, view_width: float, view_height: float, margin: float = 18.0) -> tuple[float, float, float, float]:
    width = min(width, max(1.0, view_width - 2.0 * margin))
    height = min(height, max(1.0, view_height - 2.0 * margin))
    left = min(max(x, margin), max(margin, view_width - margin - width))
    top = min(max(y, margin), max(margin, view_height - margin - height))
    return left, top, width, height


def render_equation_image(math_text: str, font_size: int = 18) -> QtGui.QImage:
    """Render a LaTeX-style mathtext fragment into a transparent QImage."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        with matplotlib.rc_context(studio_mathtext_rc()):
            width_inches = max(1.8, min(7.5, len(math_text) * 0.055))
            height_inches = max(0.42, font_size / 32.0)
            fig = Figure(figsize=(width_inches, height_inches), dpi=150)
            fig.patch.set_alpha(0.0)
            canvas = FigureCanvasAgg(fig)
            fig.text(
                0.02,
                0.50,
                math_text,
                color=(0.58, 0.96, 1.00, 1.00),
                fontsize=font_size,
                ha="left",
                va="center",
            )
            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba()).copy()
        height, width, _channels = rgba.shape
        qimg = QtGui.QImage(rgba.data, width, height, 4 * width, QtGui.QImage.Format.Format_RGBA8888)
        return qimg.copy()
    except Exception:
        return QtGui.QImage()


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def pil_to_qimage_rgba(img: Image.Image) -> QtGui.QImage:
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QtGui.QImage(data, img.width, img.height, QtGui.QImage.Format.Format_RGBA8888)
    return qimg.copy()


def create_abbe_like_mask(size: tuple[int, int] = (720, 620)) -> Image.Image:
    """Create a stylized Abbe-inspired mask without copying a source portrait."""
    width, height = size
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img, "L")
    cx, cy = int(width * 0.48), int(height * 0.47)

    draw.ellipse((cx - 100, cy - 175, cx + 100, cy + 85), fill=135)
    draw.pieslice((cx - 130, cy - 205, cx + 130, cy + 15), 185, 355, fill=95)
    draw.ellipse((cx - 135, cy - 110, cx - 88, cy + 60), fill=90)
    draw.ellipse((cx + 88, cy - 110, cx + 135, cy + 60), fill=90)
    draw.pieslice((cx - 95, cy - 25, cx + 95, cy + 140), 0, 180, fill=215)
    draw.ellipse((cx - 70, cy - 4, cx - 8, cy + 45), fill=230)
    draw.ellipse((cx + 8, cy - 4, cx + 70, cy + 45), fill=230)
    draw.polygon([(cx - 40, cy + 42), (cx + 40, cy + 42), (cx + 13, cy + 120), (cx - 13, cy + 120)], fill=235)
    draw.ellipse((cx - 60, cy - 62, cx - 12, cy - 14), outline=255, width=6)
    draw.ellipse((cx + 12, cy - 62, cx + 60, cy - 14), outline=255, width=6)
    draw.line((cx - 12, cy - 38, cx + 12, cy - 38), fill=255, width=3)
    draw.polygon([(cx - 180, cy + 125), (cx + 180, cy + 125), (cx + 250, height + 50), (cx - 250, height + 50)], fill=100)

    return img.filter(ImageFilter.GaussianBlur(0.8))


def create_microtubule_mask(size: tuple[int, int] = (720, 620), seed: int = 7) -> Image.Image:
    """Procedural microtubule-like filament network.

    Used as the default super-resolution target on the Studio launcher: the
    blinking points are sampled along these filaments, so the accumulated
    localizations resolve into a thin, crisp filament network -- the classic
    single-molecule localization microscopy demonstration image.
    """
    width, height = size
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img, "L")
    rng = np.random.default_rng(seed)
    n_filaments = 13
    step = 7.0
    for _ in range(n_filaments):
        x = rng.uniform(0.06, 0.94) * width
        y = rng.uniform(0.06, 0.94) * height
        heading = rng.uniform(0.0, 2.0 * math.pi)
        n_steps = int(rng.integers(150, 240))
        thickness = int(rng.integers(2, 5))
        brightness = int(rng.integers(155, 235))
        for _ in range(n_steps):
            heading += float(rng.normal(0.0, 0.16))
            nx = x + step * math.cos(heading)
            ny = y + step * math.sin(heading)
            draw.line((x, y, nx, ny), fill=brightness, width=thickness)
            x, y = nx, ny
            if not (0.0 <= x <= width and 0.0 <= y <= height):
                # gently steer a stray filament back toward the frame centre
                heading = math.atan2(height * 0.5 - y, width * 0.5 - x) + float(rng.normal(0.0, 0.35))
                x = min(max(x, 0.0), float(width))
                y = min(max(y, 0.0), float(height))
    return img.filter(ImageFilter.GaussianBlur(0.9))


def _bold_font(px: float):
    """A bold TrueType font. Prefers the DejaVu Sans Bold bundled with
    matplotlib so the launcher text looks identical on every platform."""
    candidates = []
    try:
        import matplotlib
        candidates.append(str(Path(matplotlib.__file__).parent
                              / "mpl-data" / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"))
    except Exception:
        pass
    candidates += ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, max(8, int(px)))
        except OSError:
            continue
    return ImageFont.load_default()


def create_text_mask(text: str = "SUPER-RESOLUTION", height_px: int = 240) -> Image.Image:
    """Render bold text into a tight grayscale sampling mask sized to the text.

    Used as the default super-resolution target on the Studio launcher: the
    blinking points are sampled across the letters, so the accumulated
    localizations resolve into the word as a wide banner.
    """
    font = _bold_font(height_px)
    measure = ImageDraw.Draw(Image.new("L", (8, 8)))
    bbox = measure.textbbox((0, 0), text, font=font)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    pad_x = int(th * 0.08)
    pad_y = int(th * 0.06)
    width, height = tw + 2 * pad_x, th + 2 * pad_y
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=255, font=font)
    return img.filter(ImageFilter.GaussianBlur(max(1.0, height_px * 0.012)))


def _filament(draw, rng, x, y, heading, n_steps, step, color, w_start, w_end, curl=0.16):
    """Draw a tapered, gently curving filament and return its list of points."""
    points = []
    span = max(1, n_steps - 1)
    for k in range(n_steps):
        heading += float(rng.normal(0.0, curl))
        nx = x + step * math.cos(heading)
        ny = y + step * math.sin(heading)
        w = w_start + (w_end - w_start) * (k / span)
        draw.line((x, y, nx, ny), fill=color, width=max(1, int(round(w))))
        x, y = nx, ny
        points.append((x, y))
    return points


def create_neuron_scene(size: tuple[int, int] = (1280, 760)) -> Image.Image:
    """Procedural colour micrograph used as the default Studio launcher target.

    A neuron -- soma plus branching dendrites, crisscrossing filaments and
    scattered puncta, in blue with magenta accents -- with the SUPER-RESOLUTION
    wordmark across the clear top band. The blinking points sample this scene
    and take their colour from it, so the launcher resolves a colour image.
    """
    width, height = size
    img = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(7)

    blue = (70, 165, 255)
    blue_dim = (38, 90, 170)
    magenta = (214, 74, 205)
    cyan = (130, 230, 255)

    # background filaments (out-of-focus neighbouring cells)
    for _ in range(9):
        _filament(draw, rng, rng.uniform(0, width), rng.uniform(height * 0.20, height),
                  rng.uniform(0, 2 * math.pi), int(rng.integers(60, 130)), 9.0,
                  blue_dim, 2.0, 1.0, curl=0.20)

    # primary dendrites radiating from the soma, each with an optional branch
    sx, sy = width * 0.24, height * 0.56
    for base_angle in np.linspace(0.0, 2.0 * math.pi, 9, endpoint=False):
        angle = float(base_angle) + float(rng.normal(0.0, 0.18))
        path = _filament(draw, rng, sx, sy, angle, int(rng.integers(80, 150)), 10.0,
                         blue, 5.5, 1.6, curl=0.13)
        if len(path) > 45 and rng.random() < 0.85:
            bi = int(len(path) * rng.uniform(0.4, 0.7))
            bx, by = path[bi]
            ba = math.atan2(by - sy, bx - sx) + float(rng.uniform(-1.1, 1.1))
            branch = _filament(draw, rng, bx, by, ba, int(rng.integers(35, 80)), 9.0,
                               blue, 2.6, 1.1, curl=0.16)
            if rng.random() < 0.5:
                for px, py in branch[::3]:
                    draw.ellipse((px - 1.7, py - 1.7, px + 1.7, py + 1.7), fill=magenta)
        if rng.random() < 0.6:
            for px, py in path[::4]:
                draw.ellipse((px - 1.9, py - 1.9, px + 1.9, py + 1.9), fill=magenta)

    # soma body: overlapping blobs, a fibrous mesh and magenta speckle
    for _ in range(7):
        rx, ry = rng.uniform(40, 80), rng.uniform(36, 70)
        ox, oy = rng.normal(0, 22), rng.normal(0, 20)
        draw.ellipse((sx + ox - rx, sy + oy - ry, sx + ox + rx, sy + oy + ry), fill=blue_dim)
    for _ in range(180):
        ang, rad = rng.uniform(0, 2 * math.pi), abs(rng.normal(0, 52))
        cx0 = sx + rad * math.cos(ang)
        cy0 = sy + rad * 0.82 * math.sin(ang)
        a2, ln = rng.uniform(0, 2 * math.pi), rng.uniform(8, 26)
        draw.line((cx0, cy0, cx0 + ln * math.cos(a2), cy0 + ln * math.sin(a2)), fill=blue, width=1)
    for _ in range(60):
        ang, rad = rng.uniform(0, 2 * math.pi), abs(rng.normal(0, 46))
        px = sx + rad * math.cos(ang)
        py = sy + rad * 0.82 * math.sin(ang)
        s = rng.uniform(1.5, 3.6)
        draw.ellipse((px - s, py - s, px + s, py + s), fill=magenta)

    # scattered puncta across the field
    for _ in range(150):
        px, py = rng.uniform(0, width), rng.uniform(height * 0.16, height)
        s = rng.uniform(1.5, 6.0)
        shade = rng.uniform(0.45, 1.0)
        draw.ellipse((px - s, py - s, px + s, py + s),
                     fill=(int(blue[0] * shade), int(blue[1] * shade), int(blue[2] * shade)))

    # SUPER-RESOLUTION wordmark across the clear top band
    text = "SUPER-RESOLUTION"
    font = _bold_font(int(height * 0.112))
    bbox = draw.textbbox((0, 0), text, font=font)
    tx = (width - (bbox[2] - bbox[0])) / 2.0 - bbox[0]
    ty = height * 0.032 - bbox[1]
    draw.text((tx, ty), text, fill=cyan, font=font)

    return img.filter(ImageFilter.GaussianBlur(0.8))


def make_scene_ghost(scene: Image.Image, blur_radius: float = 24.0) -> QtGui.QImage:
    """A soft, blurred colour version of the scene (the diffraction-limited look)."""
    blurred = scene.convert("RGB").filter(ImageFilter.GaussianBlur(blur_radius))
    arr = np.asarray(blurred, dtype=np.uint8)
    alpha = np.clip(arr.max(axis=2).astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
    rgba = np.dstack([arr, alpha])
    return pil_to_qimage_rgba(Image.fromarray(rgba, "RGBA"))


def image_to_sampling_mask(target_path: str | None, mask_size: tuple[int, int] = (720, 620)) -> Image.Image:
    if target_path is None:
        return create_text_mask()

    path = Path(target_path)
    if not path.exists():
        raise FileNotFoundError(f"Target image not found: {path}")

    src = Image.open(path).convert("L")
    src = ImageOps.autocontrast(src)
    src = ImageOps.contain(src, mask_size)
    canvas = Image.new("L", mask_size, 255)
    canvas.paste(src, ((mask_size[0] - src.width) // 2, (mask_size[1] - src.height) // 2))

    gray = ImageOps.autocontrast(canvas)
    edge = ImageOps.autocontrast(gray.filter(ImageFilter.FIND_EDGES))
    weight = 0.55 * (255.0 - np.asarray(gray, dtype=np.float32)) + 0.45 * np.asarray(edge, dtype=np.float32)
    weight = np.clip(weight, 0, 255)
    weight[weight < 25] = 0
    return Image.fromarray(weight.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(0.7))


def make_colored_ghost(mask: Image.Image, blur_radius: float = 20.0) -> QtGui.QImage:
    blurred = ImageOps.autocontrast(mask).filter(ImageFilter.GaussianBlur(blur_radius))
    arr = np.asarray(blurred, dtype=np.float32) / 255.0
    rgba = np.zeros((blurred.height, blurred.width, 4), dtype=np.uint8)
    rgba[..., 0] = (45 * arr).astype(np.uint8)
    rgba[..., 1] = (95 * arr).astype(np.uint8)
    rgba[..., 2] = (155 * arr).astype(np.uint8)
    rgba[..., 3] = (150 * arr).astype(np.uint8)
    return pil_to_qimage_rgba(Image.fromarray(rgba, "RGBA"))


class EquationGlyph:
    def __init__(self, text: str, image: QtGui.QImage, x: float, y: float, phase: float, speed: float, alpha: int, size: int):
        self.text = text
        self.image = image
        self.x = x
        self.y = y
        self.phase = phase
        self.speed = speed
        self.alpha = alpha
        self.size = size


class SuperResolutionBackground(QtWidgets.QWidget):
    def __init__(
        self,
        target_image_path: str | None = None,
        parent: QtWidgets.QWidget | None = None,
        n_points: int = 9000,
        fps: int = 30,
        loop_seconds: float = 12.0,
        hold_seconds: float = 2.6,
        fade_seconds: float = 3.8,
    ):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.n_points = int(n_points)
        self.fps = int(fps)
        self.loop_seconds = float(loop_seconds)
        self.hold_seconds = float(hold_seconds)
        self.fade_seconds = float(fade_seconds)
        self.cycle_seconds = self.loop_seconds + self.hold_seconds + self.fade_seconds
        self.rng = np.random.default_rng(11)
        self.start_time = time.perf_counter()
        self.last_frame_index = -1
        self.frozen = False

        if target_image_path is None:
            self.scene = create_neuron_scene()
            self.scene_arr = np.asarray(self.scene.convert("RGB"))
            weight = self.scene_arr.max(axis=2).astype(np.float64)
            band = int(weight.shape[0] * 0.18)
            weight[:band] *= 3.4          # the wordmark band resolves densely
            self.sample_weight = weight
            self.mask = Image.fromarray(np.clip(weight, 0, 255).astype(np.uint8), "L")
            self.ghost = make_scene_ghost(self.scene)
        else:
            self.mask = image_to_sampling_mask(target_image_path)
            self.scene_arr = None
            self.sample_weight = np.asarray(self.mask, dtype=np.float64)
            self.ghost = make_colored_ghost(self.mask)
        self._prepare_points()
        self._prepare_equations()

        self.localization_layer = QtGui.QPixmap(10, 10)
        self.localization_layer.fill(QtCore.Qt.GlobalColor.transparent)
        self._layer_size = (10, 10)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(max(1, int(1000 / self.fps)))

    def _prepare_points(self):
        arr = np.clip(self.sample_weight.astype(np.float64), 0, None)
        height, width = arr.shape
        if arr.sum() <= 0:
            raise RuntimeError("Sampling weight has no positive pixels.")

        flat = self.rng.choice(width * height, size=self.n_points, replace=True, p=(arr.ravel() / arr.sum()))
        y = flat // width
        x = flat % width
        self.point_x = (x + self.rng.normal(0, 0.7, self.n_points)) / width
        self.point_y = (y + self.rng.normal(0, 0.7, self.n_points)) / height
        self.appear_t = self.rng.beta(1.25, 1.95, self.n_points) * self.loop_seconds
        self.point_colors = []
        if self.scene_arr is not None:
            sa = self.scene_arr
            hs, ws = sa.shape[0], sa.shape[1]
            xi = np.clip(x, 0, ws - 1)
            yi = np.clip(y, 0, hs - 1)
            for i in range(self.n_points):
                r, g, b = (int(c) for c in sa[yi[i], xi[i]])
                peak = max(r, g, b)
                if peak < 95:
                    lift = 95.0 / max(1, peak)
                    r = min(255, int(r * lift))
                    g = min(255, int(g * lift))
                    b = min(255, int(b * lift))
                self.point_colors.append(QtGui.QColor(r, g, b, 210))
        else:
            for value in self.rng.random(self.n_points):
                if value < 0.62:
                    self.point_colors.append(QtGui.QColor(70, 225, 240, 150))
                elif value < 0.90:
                    self.point_colors.append(QtGui.QColor(90, 255, 190, 145))
                else:
                    self.point_colors.append(QtGui.QColor(230, 255, 255, 150))

    def _prepare_equations(self):
        self.equations: list[EquationGlyph] = []
        slots = studio_equation_layout_slots()
        style = studio_reference_style()
        speed_min, speed_max = style["equation_speed_range"]
        for i, text in enumerate(equation_fragments()):
            x, y = slots[i % len(slots)]
            self.equations.append(
                EquationGlyph(
                    text=text,
                    image=render_equation_image(text, font_size=int(self.rng.choice(studio_equation_font_sizes()))),
                    x=float(x),
                    y=float(y),
                    phase=float(self.rng.uniform(0, 2 * math.pi)),
                    speed=float(self.rng.uniform(speed_min, speed_max)),
                    alpha=int(self.rng.integers(72, 122)),
                    size=int(self.rng.choice([13, 15, 17, 19])),
                )
            )

    def reset_animation(self):
        self.start_time = time.perf_counter()
        self.last_frame_index = -1
        self.localization_layer.fill(QtCore.Qt.GlobalColor.transparent)
        self.frozen = False
        if not self.timer.isActive():
            self.timer.start(max(1, int(1000 / self.fps)))
        self.update()

    def skip_intro(self):
        self.frozen = True
        self.timer.stop()
        self.localization_layer.fill(QtCore.Qt.GlobalColor.transparent)
        self._draw_all_localizations()
        self.update()

    def _target_rect(self) -> QtCore.QRectF:
        # The colour scene maps across the whole launcher window.
        return QtCore.QRectF(0.0, 0.0, float(self.width()), float(self.height()))

    def _ensure_layer(self):
        size = (max(2, self.width()), max(2, self.height()))
        if size == self._layer_size:
            return
        self.localization_layer = QtGui.QPixmap(size[0], size[1])
        self.localization_layer.fill(QtCore.Qt.GlobalColor.transparent)
        self._layer_size = size
        self.last_frame_index = -1
        self.start_time = time.perf_counter()

    def _map_point(self, xn: float, yn: float) -> QtCore.QPointF:
        rect = self._target_rect()
        return QtCore.QPointF(rect.left() + xn * rect.width(), rect.top() + yn * rect.height())

    def _draw_all_localizations(self):
        self._ensure_layer()
        painter = QtGui.QPainter(self.localization_layer)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for i in range(self.n_points):
            pt = self._map_point(float(self.point_x[i]), float(self.point_y[i]))
            col = QtGui.QColor(self.point_colors[i])
            col.setAlpha(int(col.alpha() * 0.85))
            painter.setPen(QtGui.QPen(col, 1.6))
            painter.drawPoint(pt)
        painter.end()

    def _draw_new_localizations(self, current_t: float):
        frame_index = int(current_t * self.fps)
        if frame_index == self.last_frame_index:
            return
        start_frame = max(0, self.last_frame_index + 1)
        self.last_frame_index = frame_index
        t0 = start_frame / self.fps
        t1 = frame_index / self.fps
        idx = np.where((self.appear_t >= t0) & (self.appear_t <= t1))[0]

        painter = QtGui.QPainter(self.localization_layer)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for i in idx:
            pt = self._map_point(float(self.point_x[i]), float(self.point_y[i]))
            col = QtGui.QColor(self.point_colors[i])
            col.setAlpha(int(col.alpha() * 0.9))
            painter.setPen(QtGui.QPen(col, 1.6))
            painter.drawPoint(pt)
        painter.end()

    def paintEvent(self, event):
        self._ensure_layer()
        if self.frozen:
            elapsed = self.loop_seconds + self.hold_seconds * 0.5
        else:
            elapsed = time.perf_counter() - self.start_time
            if elapsed > self.cycle_seconds:
                self.reset_animation()
                elapsed = 0.0

        # The cycle runs build -> hold (bright) -> fade, then quietly restarts.
        build_t = min(elapsed, self.loop_seconds)
        u = build_t / self.loop_seconds
        building = elapsed < self.loop_seconds
        if building:
            layer_opacity, glow = 0.85, 0.0
        elif elapsed < self.loop_seconds + self.hold_seconds:
            hp = (elapsed - self.loop_seconds) / self.hold_seconds
            layer_opacity = 0.85 + 0.15 * smoothstep(min(1.0, hp * 2.0))
            glow = smoothstep(min(1.0, hp * 1.6))
        else:
            fp = (elapsed - self.loop_seconds - self.hold_seconds) / self.fade_seconds
            layer_opacity = 1.0 - smoothstep(min(1.0, fp))
            glow = 0.6 * (1.0 - smoothstep(min(1.0, fp)))

        if not self.frozen:
            self._draw_new_localizations(build_t)

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        self._draw_background(painter)
        self._draw_diffraction_ghost(painter, u)
        self._draw_optics_rings(painter, u)
        self._draw_equations(painter, u)
        if building and not self.frozen:
            self._draw_blinking_psfs(painter, build_t)
        if layer_opacity > 0.01:
            painter.setOpacity(layer_opacity)
            painter.drawPixmap(0, 0, self.localization_layer)
            if glow > 0.01:
                painter.setOpacity(min(1.0, glow * 0.4))
                painter.drawPixmap(0, 0, self.localization_layer)
            painter.setOpacity(1.0)
        self._draw_center_vignette(painter)
        self._draw_edge_vignette(painter)
        painter.end()

    def _draw_background(self, painter: QtGui.QPainter):
        style = studio_reference_style()
        start_color, mid_color, end_color = style["background_base_colors"]
        grad = QtGui.QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, QtGui.QColor(*start_color))
        grad.setColorAt(0.42, QtGui.QColor(*mid_color))
        grad.setColorAt(1.0, QtGui.QColor(*end_color))
        painter.fillRect(self.rect(), grad)

        center_shadow = QtGui.QRadialGradient(QtCore.QPointF(self.width() * 0.50, self.height() * 0.48), self.width() * 0.58)
        center_shadow.setColorAt(0.0, QtGui.QColor(0, 0, 0, 88))
        center_shadow.setColorAt(0.55, QtGui.QColor(3, 8, 18, 34))
        center_shadow.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), center_shadow)

        radial = QtGui.QRadialGradient(QtCore.QPointF(self.width() * 0.82, self.height() * 0.76), self.width() * 0.62)
        haze = style["background_bottom_right_haze"]
        radial.setColorAt(0.0, QtGui.QColor(*haze))
        radial.setColorAt(0.45, QtGui.QColor(12, 18, 32, 26))
        radial.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), radial)

        left_haze = QtGui.QRadialGradient(QtCore.QPointF(self.width() * 0.12, self.height() * 0.92), self.width() * 0.50)
        left_haze.setColorAt(0.0, QtGui.QColor(8, 20, 28, 28))
        left_haze.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), left_haze)

    def _draw_diffraction_ghost(self, painter: QtGui.QPainter, u: float):
        opacity = 0.42 - 0.24 * smoothstep(u / 0.35) if u < 0.35 else 0.10
        if u > 0.62:
            opacity += 0.10 * smoothstep((u - 0.62) / 0.25)
        painter.save()
        painter.setOpacity(max(0.04, opacity))
        painter.drawImage(self._target_rect(), self.ghost)
        painter.restore()

    def _draw_optics_rings(self, painter: QtGui.QPainter, u: float):
        cx = self.width() * 0.77
        cy = self.height() * 0.52
        max_r = min(self.width(), self.height()) * 0.31
        painter.save()
        for k in range(5):
            alpha = int(18 + 14 * (0.5 + 0.5 * math.sin(2 * math.pi * u + k * 0.7)))
            painter.setPen(QtGui.QPen(QtGui.QColor(90, 220, 240, alpha), 1.0))
            radius = max_r * (0.28 + k * 0.14)
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius * 0.52)
        painter.restore()

    def _draw_equations(self, painter: QtGui.QPainter, u: float):
        painter.save()
        style = studio_reference_style()
        drift_x, drift_y = style["equation_drift_px"]
        zoom_min, zoom_max = style["equation_zoom_range"]
        for eq in self.equations:
            phase = eq.phase + 2 * math.pi * u * eq.speed
            x = eq.x * self.width() + drift_x * math.sin(phase) + 0.35 * drift_x * math.sin(phase * 0.43 + 1.7)
            y = eq.y * self.height() + drift_y * math.cos(0.75 * phase) + 0.45 * drift_y * math.sin(phase * 0.51)
            opacity = (eq.alpha / 255.0) * (0.64 + 0.36 * math.sin(phase) ** 2)
            if not eq.image.isNull():
                zoom = zoom_min + (zoom_max - zoom_min) * (0.5 + 0.5 * math.sin(phase * 1.35))
                scale = max(0.38, min(0.58, self.width() / 2200.0)) * zoom
                target_width = eq.image.width() * scale
                target_height = eq.image.height() * scale
                max_width = self.width() * style["equation_max_width_fraction"]
                if target_width > max_width:
                    compact_scale = max_width / target_width
                    target_width *= compact_scale
                    target_height *= compact_scale
                left, top, width, height = clamped_equation_rect(
                    x,
                    y - target_height * 0.5,
                    target_width,
                    target_height,
                    self.width(),
                    self.height(),
                    margin=18.0,
                )
                target = QtCore.QRectF(left, top, width, height)
                glow = target.adjusted(-3.0, -3.0, 3.0, 3.0)
                painter.setOpacity(opacity * 0.28)
                painter.drawImage(glow, eq.image)
                painter.setOpacity(opacity)
                painter.drawImage(target, eq.image)
            else:
                painter.setOpacity(opacity)
                painter.setPen(QtGui.QColor(115, 230, 245, int(eq.alpha)))
                font = QtGui.QFont("Segoe UI", eq.size)
                font.setWeight(QtGui.QFont.Weight.Light)
                painter.setFont(font)
                painter.drawText(QtCore.QPointF(x, y), eq.text)
            painter.setOpacity(1.0)
        painter.restore()

    def _draw_blinking_psfs(self, painter: QtGui.QPainter, current_t: float):
        idx = np.where(np.abs(self.appear_t - current_t) <= 0.13)[0]
        if len(idx) > 85:
            idx = self.rng.choice(idx, size=85, replace=False)
        painter.save()
        halo_multiplier = studio_reference_style()["psf_halo_multiplier"]
        for i in idx:
            age = abs(float(self.appear_t[i]) - current_t) / 0.13
            alpha = int(210 * (1.0 - age) ** 1.5)
            if alpha <= 0:
                continue
            pt = self._map_point(float(self.point_x[i]), float(self.point_y[i]))
            radius = float(self.rng.uniform(5.5, 13.0))
            col = self.point_colors[i]
            cr, cg, cb = col.red(), col.green(), col.blue()
            grad = QtGui.QRadialGradient(pt, radius * halo_multiplier)
            grad.setColorAt(0.0, QtGui.QColor(min(255, cr + 150), min(255, cg + 150), min(255, cb + 150), alpha))
            grad.setColorAt(0.20, QtGui.QColor(min(255, cr + 70), min(255, cg + 70), min(255, cb + 70), int(alpha * 0.72)))
            grad.setColorAt(0.45, QtGui.QColor(cr, cg, cb, int(alpha * 0.40)))
            grad.setColorAt(1.0, QtGui.QColor(cr, cg, cb, 0))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawEllipse(pt, radius * halo_multiplier, radius * halo_multiplier)
        painter.restore()

    def _draw_center_vignette(self, painter: QtGui.QPainter):
        radial = QtGui.QRadialGradient(QtCore.QPointF(self.width() * 0.5, self.height() * 0.50), min(self.width(), self.height()) * 0.37)
        radial.setColorAt(0.0, QtGui.QColor(0, 0, 0, 100))
        radial.setColorAt(0.60, QtGui.QColor(0, 0, 0, 50))
        radial.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), radial)

    def _draw_edge_vignette(self, painter: QtGui.QPainter):
        radial = QtGui.QRadialGradient(QtCore.QPointF(self.width() * 0.5, self.height() * 0.5), self.width() * 0.72)
        radial.setColorAt(0.0, QtGui.QColor(0, 0, 0, 0))
        radial.setColorAt(0.70, QtGui.QColor(0, 0, 0, 30))
        radial.setColorAt(1.0, QtGui.QColor(0, 0, 0, 145))
        painter.fillRect(self.rect(), radial)


class GlassCardButton(QtWidgets.QPushButton):
    def __init__(self, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        self.setText(f"{title}\n{subtitle}")
        self.setMinimumSize(430, 94)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            """
            QPushButton {
                color: rgba(235, 252, 255, 245);
                background-color: rgba(8, 18, 34, 178);
                border: 1px solid rgba(98, 232, 245, 145);
                border-radius: 14px;
                padding: 16px 24px;
                font-size: 18px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: rgba(12, 32, 52, 220);
                border: 1px solid rgba(135, 250, 255, 230);
            }
            QPushButton:pressed {
                background-color: rgba(4, 15, 28, 235);
            }
            """
        )


class StudioWindow(QtWidgets.QWidget):
    def __init__(self, target_image_path: str | None = None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1280, 760)
        self.child_process: subprocess.Popen | None = None
        self.pending_workflow_script: Path | None = None
        self.transition_animation: QtCore.QPropertyAnimation | None = None

        self.background = SuperResolutionBackground(target_image_path=target_image_path, parent=self, n_points=34000)
        self.background.lower()

        self.overlay = QtWidgets.QWidget(self)
        self.overlay.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.overlay_effect = QtWidgets.QGraphicsOpacityEffect(self.overlay)
        self.overlay_effect.setOpacity(1.0)
        self.overlay.setGraphicsEffect(self.overlay_effect)
        layout = QtWidgets.QVBoxLayout(self.overlay)
        layout.setContentsMargins(28, 28, 28, 18)
        layout.setSpacing(0)

        center_layout = QtWidgets.QVBoxLayout()
        center_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        center_layout.setSpacing(20)

        title = QtWidgets.QLabel("ClassRoomSTORM Studio")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: rgba(235,252,255,240); font-size: 34px; font-weight: 600; background: transparent;")
        subtitle = QtWidgets.QLabel("Virtual and Experimental Super-Resolution")
        subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: rgba(180,230,240,205); font-size: 16px; background: transparent;")
        prompt = QtWidgets.QLabel("What would you like to do?")
        prompt.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        prompt.setStyleSheet("color: rgba(230,248,252,220); font-size: 15px; background: transparent;")

        center_layout.addWidget(title)
        center_layout.addWidget(subtitle)
        center_layout.addSpacing(8)
        center_layout.addWidget(prompt)

        for workflow in studio_workflows():
            button = GlassCardButton(workflow["title"], workflow["subtitle"])
            button.clicked.connect(lambda _checked=False, key=workflow["key"]: self.launch_workflow(key))
            center_layout.addWidget(button)

        reader_row = QtWidgets.QHBoxLayout()
        readers = studio_reader_links()
        for index, reader in enumerate(readers):
            button = QtWidgets.QPushButton(reader["label"])
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet("color: rgba(210,240,245,215); background: transparent; border: none; padding: 8px 12px;")
            button.clicked.connect(lambda _checked=False, key=reader["key"]: self.open_reader(key))
            reader_row.addWidget(button)
            if index < len(readers) - 1:
                separator = QtWidgets.QLabel("|")
                separator.setStyleSheet("color: rgba(160,205,215,125); background: transparent;")
                reader_row.addWidget(separator)
        reader_row.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        center_layout.addLayout(reader_row)

        credit = QtWidgets.QLabel()
        credit.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        credit.setText(studio_footer_text())
        credit.setStyleSheet("color: rgba(160,205,215,135); font-size: 11px; background: transparent;")
        self.footer_label = credit

        layout.addStretch(1)
        layout.addLayout(center_layout)
        layout.addStretch(1)
        layout.addWidget(self.footer_label)

    def resizeEvent(self, event):
        self.background.setGeometry(self.rect())
        self.overlay.setGeometry(self.rect())
        super().resizeEvent(event)

    def launch_workflow(self, key: str):
        workflow = next((item for item in studio_workflows() if item["key"] == key), None)
        if workflow is None:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Unknown workflow: {key}")
            return
        script = Path(workflow["script"])
        if not script.exists():
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Workflow script not found:\n{script}")
            return
        self.begin_workflow_transition(script)

    def begin_workflow_transition(self, script: Path):
        self.pending_workflow_script = script
        self.setEnabled(False)
        self.transition_animation = QtCore.QPropertyAnimation(self.overlay_effect, b"opacity", self)
        self.transition_animation.setDuration(STUDIO_TRANSITION_MS)
        self.transition_animation.setStartValue(1.0)
        self.transition_animation.setEndValue(0.0)
        self.transition_animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self.transition_animation.finished.connect(self.finish_workflow_transition)
        self.transition_animation.start(QtCore.QAbstractAnimation.DeletionPolicy.KeepWhenStopped)

    def finish_workflow_transition(self):
        if self.pending_workflow_script is None:
            self.setEnabled(True)
            return
        self.child_process = subprocess.Popen([sys.executable, str(self.pending_workflow_script)], cwd=str(PROJECT_ROOT))
        QtWidgets.QApplication.instance().quit()

    def open_reader(self, key: str):
        reader = next((item for item in studio_reader_links() if item["key"] == key), None)
        if reader is None:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Unknown reader: {key}")
            return
        path = reader_display_path(reader)
        if not path.exists():
            QtWidgets.QMessageBox.information(self, APP_NAME, f"{reader['label']} document will be added soon.")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))


def run_gui(target_image_path: str | None = None) -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = StudioWindow(target_image_path=target_image_path)
    window.show()
    return app.exec()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} {APP_VERSION}")
    parser.add_argument("--target", default=None, help="Optional portrait/logo image used as the animated localization target.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return run_gui(target_image_path=args.target)


if __name__ == "__main__":
    raise SystemExit(main())
