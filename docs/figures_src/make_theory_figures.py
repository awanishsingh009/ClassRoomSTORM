"""
Generate Theory document figures T1-T9 for ClassRoomSTORM V2.

Most figures are created at their final PDF width (see style.figsize), so
LaTeX inserts them at scale 1.0 and figure text is uniform across the document.
T6 and T7 are deliberately rendered on a smaller canvas and inserted at MAGx,
which enlarges their text and line work by ~50% (see the MAG constant).
T10 is produced by make_diagrams.py and T11 by make_result_figures.py.

Output: ../figures/theory/
Run:    python make_theory_figures.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import (airy_intensity, gaussian2d, thompson_se, centroid_threshold,
                   figsize, COL_RAW, COL_RECON, COL_TRUTH, COL_AXIS,
                   COL_ACCENT, PALETTE)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "figures", "theory")
R1 = 3.8317059702075125 / (2.0 * np.pi)   # first Airy zero (~0.610)

# T6 and T7 are rendered on a 1/MAG canvas and inserted at their full target
# width, so LaTeX magnifies them MAGx. This enlarges every element -- text,
# line widths and markers -- together, so the labels grow ~50% without
# unbalancing the line weights or marker sizes.
MAG = 1.5


def _no_spines(ax):
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- T1  (1.00)
def fig_T1():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=figsize(1.00, 2.65))
    span, n = 2.4, 401
    xx = np.linspace(-span, span, n)
    X, Y = np.meshgrid(xx, xx)
    im = axa.imshow(airy_intensity(np.hypot(X, Y)), origin="lower",
                    extent=(-span, span, -span, span), cmap="inferno",
                    norm=PowerNorm(0.40))
    axa.set_xlabel(r"$x\ (\lambda/\mathrm{NA})$")
    axa.set_ylabel(r"$y\ (\lambda/\mathrm{NA})$")
    axa.set_title("Airy point-spread function")
    cb = fig.colorbar(im, ax=axa, fraction=0.046, pad=0.04)
    cb.set_label(r"$I/I_0$")
    cb.ax.tick_params(labelsize=8)
    style.panel_label(axa, "a")

    r = np.linspace(0, span, 600)
    Ia = airy_intensity(r)
    below = np.where(Ia <= 0.5)[0]
    r_half = r[below[0]] if below.size else 0.25
    sig = r_half / np.sqrt(2.0 * np.log(2.0))
    axb.plot(r, Ia, color=COL_AXIS, label="Airy")
    axb.plot(r, np.exp(-0.5 * (r / sig) ** 2), color=COL_ACCENT, ls="--",
             label="Gaussian")
    axb.axhline(0.5, color="0.75", lw=0.7, ls=":")
    axb.set_xlabel(r"$r\ (\lambda/\mathrm{NA})$")
    axb.set_ylabel(r"$I/I_0$")
    axb.set_title("Gaussian approximation of the core")
    axb.set_xlim(0, span)
    axb.legend()
    style.panel_label(axb, "b")
    fig.tight_layout(w_pad=1.4)
    style.save(fig, OUT, "T1_airy_gaussian")


# ---------------------------------------------------------------- T2  (1.00)
def fig_T2():
    fig, axes = plt.subplots(1, 3, figsize=figsize(1.00, 2.2), sharey=True)
    x = np.linspace(-2.2, 2.2, 2400)
    cases = [(1.7 * R1, "resolved"), (R1, "Rayleigh limit"),
             (0.55 * R1, "unresolved")]
    for ax, (d, label), lab in zip(axes, cases, "abc"):
        s1 = airy_intensity(np.abs(x - d / 2))
        s2 = airy_intensity(np.abs(x + d / 2))
        ssum = s1 + s2
        ssum /= ssum.max()
        ax.plot(x, ssum, color=COL_AXIS, lw=1.4, label="observed")
        ax.plot(x, 0.5 * s1 / s1.max(), color=PALETTE[0], ls=":", lw=1.0)
        ax.plot(x, 0.5 * s2 / s2.max(), color=PALETTE[0], ls=":", lw=1.0,
                label="emitters")
        ax.set_title(r"%s ($d=%.2f\,\lambda/\mathrm{NA}$)" % (label, d))
        ax.set_xlabel(r"position $\ (\lambda/\mathrm{NA})$")
        ax.set_ylim(0, 1.13)
        style.panel_label(ax, lab)
    axes[0].set_ylabel(r"$I/I_{\max}$")
    axes[2].legend(loc="upper right", fontsize=7)
    fig.tight_layout(w_pad=1.0)
    style.save(fig, OUT, "T2_rayleigh")


# ---------------------------------------------------------------- T3  (1.00)
def fig_T3():
    n = 260
    yy, xx = np.indices((n, n))
    cx = cy = (n - 1) / 2.0
    Rg = np.hypot(xx - cx, yy - cy)
    obj = ((Rg > n * 0.30) & (Rg < n * 0.335)).astype(float)
    obj[int(cy), int(cx - n * 0.15):int(cx + n * 0.15)] = 1.0
    sig = n * 0.030
    psf = gaussian2d(xx - cx, yy - cy, sigma=sig)
    img = gaussian_filter(obj, sig)

    fig = plt.figure(figsize=figsize(1.00, 2.35))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.28, 1, 0.28, 1], wspace=0.06)
    img_axes = [fig.add_subplot(gs[0, k]) for k in (0, 2, 4)]
    op_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3])]
    panels = [(obj, r"object $f(x,y)$", "a"),
              (psf, r"PSF $h(x,y)$", "b"),
              (img, r"image $g=h\ast f$", "c")]
    for ax, (data, ttl, lab) in zip(img_axes, panels):
        ax.imshow(data, origin="lower", cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl)
        style.panel_label(ax, lab)
    for axo, sym in zip(op_axes, (r"$\ast$", r"$=$")):
        axo.axis("off")
        axo.text(0.5, 0.5, sym, ha="center", va="center", fontsize=15,
                 transform=axo.transAxes)
    style.save(fig, OUT, "T3_image_formation")


# ---------------------------------------------------------------- T4  (0.90)
def fig_T4():
    rng = np.random.default_rng(3)
    n = 240
    yy, xx = np.indices((n, n))
    cx = cy = (n - 1) / 2.0
    n_em = 64
    ang = np.linspace(0, 2 * np.pi, n_em, endpoint=False)
    ex = cx + n * 0.30 * np.cos(ang)
    ey = cy + n * 0.30 * np.sin(ang)
    sig = n * 0.022
    all_on = np.zeros((n, n))
    for x0, y0 in zip(ex, ey):
        all_on += gaussian2d(xx, yy, x0, y0, sig)
    sparse = np.zeros((n, n))
    for i in rng.choice(n_em, size=6, replace=False):
        sparse += gaussian2d(xx, yy, ex[i], ey[i], sig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize(0.90, 3.0))
    for ax, data, ttl, lab in [(a1, all_on, "all emitters on at once", "a"),
                               (a2, sparse, "one sparse blinking frame", "b")]:
        ax.imshow(data, origin="lower", cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl)
        style.panel_label(ax, lab)
    fig.tight_layout(w_pad=1.2)
    style.save(fig, OUT, "T4_blinking")


# ---------------------------------------------------------------- T5  (1.00)
def fig_T5():
    rng = np.random.default_rng(11)
    n = 25
    yy, xx = np.indices((n, n))
    c = (n - 1) / 2.0
    sigma, N, bg = 2.3, 280.0, 3.0
    expected = bg + gaussian2d(xx, yy, c, c, sigma, amp=N / (2 * np.pi * sigma ** 2))

    fig = plt.figure(figsize=figsize(1.00, 3.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.55,
                          wspace=0.28)
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(rng.poisson(expected), origin="lower", cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("noisy frame %d" % (i + 1))
        if i == 0:
            style.panel_label(ax, "a")

    ests = []
    for _ in range(800):
        noisy = rng.poisson(expected).astype(float)
        w = np.clip(noisy - bg, 0.0, None)
        s = w.sum()
        ests.append((xx * w).sum() / s if s > 0 else c)
    ests = np.array(ests) - c
    axh = fig.add_subplot(gs[1, :])
    axh.hist(ests, bins=34, color=COL_RECON, alpha=0.85, edgecolor="white", lw=0.3)
    axh.axvline(0.0, color=COL_TRUTH, lw=1.5, label="true position")
    axh.set_xlabel(r"position error $\hat{x}-x_0$ (pixels)")
    axh.set_ylabel("count")
    axh.set_title(r"estimated position from many noisy frames "
                  r"(s.d.\ $\approx %.3f$ px)" % ests.std())
    axh.legend()
    style.panel_label(axh, "b")
    style.save(fig, OUT, "T5_poisson_noise")


# ----------------------------------------------------- T6  (0.70, magnified MAGx)
def fig_T6():
    # Rendered small and inserted at 0.70\linewidth so LaTeX magnifies it MAGx:
    # text, lines and markers grow together and stay visually balanced.
    w, h = figsize(0.70, 3.2)
    fig, ax = plt.subplots(figsize=(w / MAG, h / MAG))
    N = np.logspace(2, 4.5, 240)
    sigma, a = 1.5, 1.0
    for b, col in zip([0.0, 2.0, 8.0], PALETTE[:3]):
        ax.loglog(N, thompson_se(N, sigma, a, b), color=col,
                  label=r"$b=%g$" % b)
    # Slope guide in the empty band below the lowest (b=0) curve, parallel to
    # it; placed mid-plot so it is clear of both the curves and the legend.
    ax.text(1.0e3, thompson_se(1.0e3, sigma, a, 0.0) * 0.60,
            r"$\mathrm{SE}\propto N^{-1/2}$", color="0.45", fontsize=8,
            rotation=-27, ha="center", va="center")
    ax.set_xlabel(r"signal photons $N$")
    ax.set_ylabel("localization precision (pixels)")
    ax.set_title("localization precision vs.\\ photon count")
    # Compact legend pinned to the empty top-right corner: short labels keep
    # the box narrow, the title carries the meaning of b, the white box keeps
    # it crisp if a curve grazes the corner.
    ax.legend(loc="upper right", title="background", frameon=True,
              facecolor="white", edgecolor="0.7", framealpha=0.92)
    fig.tight_layout()
    style.save(fig, OUT, "T6_precision_vs_photons")


# ----------------------------------------------------- T7  (0.70, magnified MAGx)
def fig_T7():
    rng = np.random.default_rng(5)
    n = 27
    yy, xx = np.indices((n, n))
    x_true, y_true = 14.4, 12.6
    sigma, N, bg = 2.4, 420.0, 4.0
    expected = bg + gaussian2d(xx, yy, x_true, y_true, sigma,
                               amp=N / (2 * np.pi * sigma ** 2))
    frame = rng.poisson(expected).astype(float)
    mask = frame >= np.quantile(frame, 0.995)
    cx, cy = centroid_threshold(frame, 0.995)

    # See fig_T6: rendered small and inserted at 1.5x to enlarge the text ~50%.
    w, h = figsize(0.70, 3.9)
    fig, ax = plt.subplots(figsize=(w / MAG, h / MAG))
    im = ax.imshow(frame, origin="lower", cmap="gray")
    ax.contour(mask.astype(float), levels=[0.5], colors=[COL_RECON],
               linewidths=1.3)
    ax.plot(x_true, y_true, marker="x", ms=8, mew=1.8, color=COL_TRUTH,
            label="true position")
    ax.plot(cx, cy, marker="o", ms=7, mfc="none", mew=1.8, color=COL_RECON,
            label="estimated centroid")
    ax.set_xlabel(r"$x$ (pixels)")
    ax.set_ylabel(r"$y$ (pixels)")
    ax.set_title("centroid of the thresholded spot")
    # Opaque white box so the labels stay readable over the dark image.
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, facecolor="white",
              edgecolor="0.7", framealpha=0.92)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("counts")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    style.save(fig, OUT, "T7_centroid")


# ---------------------------------------------------------------- T8  (0.72)
def fig_T8():
    rng = np.random.default_rng(8)
    specs = [("high precision, high accuracy", 0.10, (0.00, 0.00), "a"),
             ("high precision, low accuracy", 0.10, (0.52, 0.34), "b"),
             ("low precision, high accuracy", 0.30, (0.00, 0.00), "c"),
             ("low precision, low accuracy", 0.30, (0.52, 0.34), "d")]
    # Four genuinely square panels placed by hand. Manual placement keeps the
    # gap between every title and its panel small and identical, and removes
    # the uneven whitespace that set_aspect('equal') leaves in a subplot grid.
    W = 0.72 * style.TEXTWIDTH
    gx, gy = 0.26, 0.40                       # gaps between columns / rows (in)
    mL = mR = mB = 0.05
    mT = 0.26
    s = (W - mL - mR - gx) / 2.0              # square panel side (inches)
    H = mB + 2.0 * s + gy + mT
    fig = plt.figure(figsize=(W, H))
    for k, (ttl, spread, off, lab) in enumerate(specs):
        col, row = k % 2, k // 2
        x = mL + col * (s + gx)
        y = mB + (s + gy if row == 0 else 0.0)
        ax = fig.add_axes([x / W, y / H, s / W, s / H])
        for rr in (0.30, 0.60, 0.90):
            ax.add_patch(Circle((0, 0), rr, fill=False, color="0.84", lw=0.7))
        px = rng.normal(off[0], spread, 60)
        py = rng.normal(off[1], spread, 60)
        ax.scatter(px, py, s=11, color=COL_RECON, alpha=0.75,
                   edgecolor="white", lw=0.3)
        ax.plot(0, 0, marker="+", ms=11, mew=1.6, color=COL_TRUTH)
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.55, 1.55)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        _no_spines(ax)
        ax.set_title(ttl, pad=4)
        style.panel_label(ax, lab)
    style.save(fig, OUT, "T8_precision_accuracy")


# ---------------------------------------------------------------- T9  (0.86)
def fig_T9():
    rng = np.random.default_rng(2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize(0.86, 2.95),
                                 sharex=True, sharey=True)
    circ = np.linspace(0, 2 * np.pi, 400)
    for ax, n_loc, ttl, lab in [(a1, 16, "too few localizations", "a"),
                                (a2, 320, "enough localizations", "b")]:
        ax.plot(np.cos(circ), np.sin(circ), color="0.78", lw=1.0,
                label="true structure")
        ang = rng.uniform(0, 2 * np.pi, n_loc)
        r = 1.0 + rng.normal(0, 0.035, n_loc)
        ax.scatter(r * np.cos(ang), r * np.sin(ang), s=12, color=COL_RECON,
                   alpha=0.8, edgecolor="white", lw=0.3, label="localizations")
        ax.set_aspect("equal")
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_xticks([]); ax.set_yticks([])
        _no_spines(ax)
        ax.set_title(ttl)
        style.panel_label(ax, lab)
    a1.legend(loc="lower center", fontsize=7.5, ncol=2,
              bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(w_pad=1.2)
    style.save(fig, OUT, "T9_nyquist")


def main():
    style.setup()
    print("Generating Theory figures T1-T9 into", os.path.normpath(OUT))
    for fn in (fig_T1, fig_T2, fig_T3, fig_T4, fig_T5,
               fig_T6, fig_T7, fig_T8, fig_T9):
        fn()
    print("Theory figures done.")


if __name__ == "__main__":
    main()
