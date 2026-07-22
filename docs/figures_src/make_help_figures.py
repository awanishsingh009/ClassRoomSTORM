"""
Generate Help document figures H4, H5, H8 for ClassRoomSTORM V2.

Each figure is created at its final PDF width (style.figsize), so LaTeX
inserts it at scale 1.0 and all figure text is uniform. The flow diagrams
H1 and H9 are produced by make_diagrams.py; the GUI screenshots
(H2, H3, H6, H7) are captured by make_screenshots.py.

Output: ../figures/help/
Run:    python make_help_figures.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import gaussian2d, figsize, COL_RAW, COL_RECON, COL_ACCENT

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "figures", "help")


# ---------------------------------------------------------------- H4  (0.80)
def fig_H4():
    """A source pattern becomes a set of source points (two equal panels)."""
    n = 120
    yy, xx = np.indices((n, n))
    c = (n - 1) / 2.0
    ang = np.linspace(0, 2 * np.pi, 18, endpoint=False)
    px = np.append(c + n * 0.30 * np.cos(ang), c)
    py = np.append(c + n * 0.30 * np.sin(ang), c)
    mask = np.zeros((n, n))
    for x0, y0 in zip(px, py):
        mask[np.hypot(xx - x0, yy - y0) < 2.3] = 1.0

    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize(0.80, 2.7))
    a1.imshow(mask, origin="lower", cmap="gray", extent=(0, n, 0, n))
    a1.set_title("source pattern (mask)")
    a2.scatter(px, py, s=22, color=COL_RECON, edgecolor="white", lw=0.5)
    a2.set_title("source points")
    a2.set_xlim(0, n); a2.set_ylim(0, n)
    for ax, lab in ((a1, "a"), (a2, "b")):
        ax.set_box_aspect(1)            # identical square panels
        ax.set_xticks([]); ax.set_yticks([])
        style.panel_label(ax, lab)
    for s in a2.spines.values():        # light frame to balance the mask panel
        s.set_visible(True); s.set_color("0.75"); s.set_linewidth(0.6)
    fig.tight_layout(w_pad=1.6)
    style.save(fig, OUT, "H4_pattern_preview")


# ---------------------------------------------------------------- H5  (0.70)
def fig_H5():
    """A single simulated blinking frame."""
    rng = np.random.default_rng(7)
    H, W = 80, 104
    yy, xx = np.indices((H, W))
    clean = np.full((H, W), 2.0)
    for _ in range(4):
        x0 = rng.uniform(16, W - 16)
        y0 = rng.uniform(16, H - 16)
        clean += gaussian2d(xx, yy, x0, y0, 4.0, amp=rng.uniform(11.0, 16.0))
    frame = rng.poisson(clean)

    fig, ax = plt.subplots(figsize=figsize(0.70, 3.4))
    im = ax.imshow(frame, origin="lower", cmap="gray")
    ax.set_title("a single simulated blinking frame")
    ax.set_xlabel(r"$x$ (pixels)")
    ax.set_ylabel(r"$y$ (pixels)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("counts")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    style.save(fig, OUT, "H5_blinking_frame")


# ---------------------------------------------------------------- H8  (1.00)
def fig_H8():
    """Linked profile viewer schematic: red = raw, blue = reconstruction."""
    n = 64
    yy, xx = np.indices((n, n))
    c = (n - 1) / 2.0
    raw_img = gaussian2d(xx, yy, c, c, 8.5)
    rec_img = gaussian2d(xx, yy, c, c, 2.2)

    fig = plt.figure(figsize=figsize(1.00, 4.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.45,
                          wspace=0.22)
    ar = fig.add_subplot(gs[0, 0])
    ac = fig.add_subplot(gs[0, 1])
    ap = fig.add_subplot(gs[1, :])

    for ax, img, ttl, lab in [(ar, raw_img, "raw mean", "a"),
                              (ac, rec_img, "reconstruction", "b")]:
        ax.imshow(img, origin="lower", cmap="gray")
        ax.axhline(c, color=COL_ACCENT, lw=1.0, ls=(0, (4, 2)))
        ax.axvline(c, color=COL_ACCENT, lw=1.0, ls=(0, (4, 2)))
        ax.set_box_aspect(1)
        ax.set_title(ttl)
        ax.set_xticks([]); ax.set_yticks([])
        style.panel_label(ax, lab)

    x = np.arange(n)
    ap.plot(x, raw_img[int(c), :] / raw_img.max(), color=COL_RAW, label="raw")
    ap.plot(x, rec_img[int(c), :] / rec_img.max(), color=COL_RECON,
            label="reconstruction")
    ap.set_xlabel(r"pixel coordinate")
    ap.set_ylabel(r"normalized intensity")
    ap.set_title("linked line profile at the selected point")
    ap.set_xlim(0, n - 1)
    ap.legend(loc="upper right")
    style.panel_label(ap, "c")
    style.save(fig, OUT, "H8_linked_profile")


def main():
    style.setup()
    print("Generating Help figures H4, H5, H8 into", os.path.normpath(OUT))
    for fn in (fig_H4, fig_H5, fig_H8):
        fn()
    print("Help figures done.")


if __name__ == "__main__":
    main()
