"""
Shared plotting style, figure-sizing policy, and physics helpers for the
ClassRoomSTORM V1.2 documentation figures.

SIZING POLICY
-------------
Every figure is designed at its *final* width in the PDF, so LaTeX inserts it
at scale 1.0 and the font sizes set here are exactly the font sizes that appear
on the page. The companion documents have a text width of TEXTWIDTH inches (a4 paper,
24 mm margins). A figure that will be inserted at `width=f\linewidth` must be
created with `figsize=(f*TEXTWIDTH, height)`.

Text is typeset with LaTeX (text.usetex) when available, with an automatic
Computer-Modern-mathtext fallback, so figure typography matches the documents.
"""
import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import j1, erf

# Text width of the companion LaTeX documents, in inches (a4, 24 mm margins).
TEXTWIDTH = 162.0 / 25.4

# ----------------------------------------------------------------------
# Colour convention
#   red  -> raw / widefield / diffraction-limited data
#   blue -> reconstruction / super-resolution data
#   dark grey -> axes, ticks, text, annotation arrows
# ----------------------------------------------------------------------
COL_RAW    = "#c1272d"
COL_RECON  = "#0072B2"
COL_TRUTH  = "#39775A"
COL_AXIS   = "#20252C"
COL_ACCENT = "#D55E00"
PALETTE    = ["#1f5f9e", "#e6820a", "#2e7d32", "#c1272d", "#7b3294", "#4ba3c3"]

PDF_META = {
    "Author":  "Awanish Pratap Singh",
    "Title":   "ClassRoomSTORM documentation figure",
    "Subject": "Localization-based super-resolution microscopy teaching figure",
}

_USETEX = False


def figsize(width_frac, height_in):
    """Figure size for a figure inserted at `width_frac` * text width.

    width_frac : fraction of \\linewidth the figure is inserted at
    height_in  : figure height in inches
    """
    return (width_frac * TEXTWIDTH, float(height_in))


def _probe_usetex():
    try:
        with plt.rc_context({"text.usetex": True, "font.family": "serif",
                             "text.latex.preamble": r"\usepackage{amsmath,lmodern}"}):
            fig = plt.figure(figsize=(1.4, 1.0))
            fig.text(0.5, 0.5, r"$x\;y\;\lambda\;\sigma\;\mathrm{NA}$")
            probe = os.path.join(tempfile.gettempdir(), "__crs_usetex_probe.png")
            fig.savefig(probe, dpi=60)
            plt.close(fig)
            try:
                os.remove(probe)
            except OSError:
                pass
        return True
    except Exception:
        plt.close("all")
        return False


def setup():
    """Apply the shared style. Font sizes here are final PDF sizes (scale 1.0)."""
    global _USETEX
    _USETEX = _probe_usetex()
    rc = {
        "figure.dpi":        150,
        "savefig.dpi":       400,
        "savefig.bbox":      None,
        "savefig.pad_inches": 0.015,
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
        # --- final-PDF font sizes (uniform across every figure) ---
        "font.size":         9,
        "axes.titlesize":    9,
        "axes.titleweight":  "normal",
        "axes.titlecolor":   COL_AXIS,
        "axes.labelsize":    9,
        "axes.labelcolor":   COL_AXIS,
        "axes.edgecolor":    COL_AXIS,
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "text.color":        COL_AXIS,
        "lines.linewidth":   1.3,
        "legend.fontsize":   8.5,
        "legend.frameon":    False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "xtick.color":       COL_AXIS,
        "ytick.color":       COL_AXIS,
        "xtick.labelcolor":  COL_AXIS,
        "ytick.labelcolor":  COL_AXIS,
        "xtick.labelsize":   8.5,
        "ytick.labelsize":   8.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  2.6,
        "ytick.major.size":  2.6,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "image.cmap":        "viridis",
    }
    if _USETEX:
        rc.update({
            "text.usetex":         True,
            "font.family":         "serif",
            "font.serif":          ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath,lmodern}",
        })
    else:
        rc.update({
            "text.usetex":      False,
            "font.family":      "serif",
            "font.serif":       ["CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        })
    plt.rcParams.update(rc)
    print("  text rendering: %s" % ("LaTeX (text.usetex)" if _USETEX
                                    else "Computer Modern mathtext (fallback)"))
    return _USETEX


def using_tex():
    return _USETEX


def save(fig, outdir, name):
    """Save a figure as vector PDF and 400-dpi PNG."""
    # Attach the panel marker to its title, once the final title is known.
    # A separate floating letter can collide with a two-line title or legend.
    for ax in fig.axes:
        letter = getattr(ax, "_scientific_panel", None)
        if letter:
            title = ax.get_title()
            ax.set_title("")
            prefix = r"\textbf{(%s)}" % letter if _USETEX else "(%s)" % letter
            ax.set_title(prefix + " " + title, loc="left", fontsize=9, pad=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = list(fig.texts)
    for ax in fig.axes:
        labels.extend([ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label])
        if ax.get_legend():
            labels.extend(ax.get_legend().get_texts())
    for legend in fig.legends:
        labels.extend(legend.get_texts())
    # Preserve physical width: a clipped label must be reworded or repositioned,
    # never hidden by bbox='tight', which silently changes the printed scale.
    for label in labels:
        if label.get_visible() and label.get_text():
            bounds = label.get_window_extent(renderer)
            if (bounds.x0 < -.5 or bounds.y0 < -.5 or
                    bounds.x1 > fig.bbox.width + .5 or bounds.y1 > fig.bbox.height + .5):
                raise ValueError(f"{name}: text crosses the fixed canvas: {label.get_text()!r}")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".pdf"), metadata=PDF_META)
    fig.savefig(os.path.join(outdir, name + ".png"))
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(os.path.join(outdir, name + ".svg"))
    plt.close(fig)
    print("  wrote " + name + ".pdf / .png")


def panel_label(ax, letter, x=-0.015, y=1.02):
    """Register a bold (a)-style prefix for the final panel title.

    x/y remain accepted for older generators; positioning is shared by save().
    """
    ax._scientific_panel = letter


# ----------------------------------------------------------------------
# Physics helpers
# ----------------------------------------------------------------------
def airy_intensity(R, alpha=2.0 * np.pi):
    R = np.asarray(R, dtype=float)
    Z = alpha * R
    out = np.ones_like(Z)
    nz = Z != 0
    out[nz] = (2.0 * j1(Z[nz]) / Z[nz]) ** 2
    return out


def gaussian2d(X, Y, x0=0.0, y0=0.0, sigma=1.0, amp=1.0):
    return amp * np.exp(-0.5 * (((X - x0) / sigma) ** 2 + ((Y - y0) / sigma) ** 2))


def gaussian1d(x, x0=0.0, sigma=1.0, amp=1.0):
    return amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def pixel_integrated_gaussian_1d(j_idx, x0, sigma, N, b=0.0):
    j_idx = np.asarray(j_idx, dtype=float)
    lo = (j_idx - 0.5 - x0) / (np.sqrt(2.0) * sigma)
    hi = (j_idx + 0.5 - x0) / (np.sqrt(2.0) * sigma)
    return b + N * 0.5 * (erf(hi) - erf(lo))


def thompson_se(N, sigma, a=1.0, b=0.0):
    """Approximate single-coordinate precision; b is background RMS noise.

    sigma and a must use the same length unit; N and b are photon-equivalent.
    This is a theoretical reference, not the software centroid uncertainty.
    """
    N = np.asarray(N, dtype=float)
    return np.sqrt((sigma ** 2 + a ** 2 / 12.0) / N
                   + (8.0 * np.pi * sigma ** 4 * b ** 2) / (a ** 2 * N ** 2))


def centroid_threshold(frame, q=0.995):
    """Use the maintained application estimator for teaching examples."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from classroomstorm_core.reconstruction import _localize_single
    loc = _localize_single(frame, q, 0)
    if loc is None:
        return float("nan"), float("nan")
    return loc["x_px"], loc["y_px"]
