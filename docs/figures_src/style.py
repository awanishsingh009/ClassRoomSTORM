"""
Shared plotting style, figure-sizing policy, and physics helpers for the
ClassRoomSTORM V2 documentation figures.

SIZING POLICY
-------------
Every figure is designed at its *final* width in the PDF, so LaTeX inserts it
at scale 1.0 and the font sizes set here are exactly the font sizes that appear
on the page. The V2 documents have a text width of TEXTWIDTH inches (a4 paper,
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

# Text width of the V2 LaTeX documents, in inches (a4, 24 mm margins).
TEXTWIDTH = 6.38

# ----------------------------------------------------------------------
# Colour convention
#   red  -> raw / widefield / diffraction-limited data
#   blue -> reconstruction / super-resolution data
#   dark grey -> axes, ticks, text, annotation arrows
# ----------------------------------------------------------------------
COL_RAW    = "#c1272d"
COL_RECON  = "#1f5f9e"
COL_TRUTH  = "#2e7d32"
COL_AXIS   = "#333333"
COL_ACCENT = "#e6820a"
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
    return (round(width_frac * TEXTWIDTH, 3), float(height_in))


def _probe_usetex():
    try:
        with plt.rc_context({"text.usetex": True, "font.family": "serif",
                             "text.latex.preamble": r"\usepackage{amsmath}"}):
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
        "savefig.bbox":      "tight",
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
        "legend.fontsize":   8,
        "legend.frameon":    False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "xtick.color":       COL_AXIS,
        "ytick.color":       COL_AXIS,
        "xtick.labelcolor":  COL_AXIS,
        "ytick.labelcolor":  COL_AXIS,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
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
            "text.latex.preamble": r"\usepackage{amsmath}",
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
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".pdf"), metadata=PDF_META)
    fig.savefig(os.path.join(outdir, name + ".png"))
    plt.close(fig)
    print("  wrote " + name + ".pdf / .png")


def panel_label(ax, letter, x=-0.015, y=1.02):
    """Place one consistent bold panel label just outside the top-left corner.

    Bold lower-case letter, no parentheses; same placement on every figure.
    """
    txt = r"\textbf{%s}" % letter if _USETEX else r"$\mathbf{%s}$" % letter
    ax.text(x, y, txt, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=COL_AXIS)


# ----------------------------------------------------------------------
# Physics helpers
# ----------------------------------------------------------------------
def airy_intensity(R, alpha=2.0 * np.pi):
    R = np.asarray(R, dtype=float)
    Z = alpha * R
    out = np.ones_like(Z)
    nz = Z != 0
    out[nz] = (2.0 * j1(Z[nz]) / Z[nz]) ** 2
    m = out.max()
    return out / m if m > 0 else out


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
    N = np.asarray(N, dtype=float)
    return np.sqrt((sigma ** 2 + a ** 2 / 12.0) / N
                   + (8.0 * np.pi * sigma ** 4 * b ** 2) / (a ** 2 * N ** 2))


def centroid_threshold(frame, q=0.995):
    """ClassRoomSTORM V1.1 single-emitter localization (threshold + centroid)."""
    frame = np.asarray(frame, dtype=float)
    thr = np.quantile(frame, q)
    mask = frame >= thr
    if not mask.any():
        iy, ix = np.unravel_index(int(np.argmax(frame)), frame.shape)
        return float(ix), float(iy)
    ys, xs = np.indices(frame.shape)
    w = frame * mask
    s = w.sum()
    return float((xs * w).sum() / s), float((ys * w).sum() / s)
