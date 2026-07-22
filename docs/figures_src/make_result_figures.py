"""
Generate the reconstruction-result figure T11 for ClassRoomSTORM V2.

A short blinking video is simulated and reconstructed with the ClassRoomSTORM
V1.1 pipeline (threshold at the 0.995 intensity quantile, intensity-weighted
centroid per bright component, Gaussian-splat rendering). The figure is created
at full text width (scale 1.0 in the PDF); the three panels share one height.

Output: ../figures/theory/
Run:    python make_result_figures.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import gaussian2d, figsize, COL_RAW, COL_RECON, COL_ACCENT

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "figures", "theory")

RENDER_SIGMA_PX = 1.5
THRESHOLD_Q = 0.995


def simulate_blinking_video():
    """Two closely spaced rows of emitters seen through a wide PSF."""
    rng = np.random.default_rng(2026)
    H = W = 96
    yy, xx = np.indices((H, W))
    xs = np.linspace(17, W - 17, 16)
    rows = [H / 2.0 - 7.0, H / 2.0 + 7.0]
    emitters = [(x, y) for y in rows for x in xs]
    sigma_psf = 5.4              # wide PSF: the two rows merge in the raw image
    background = 2.0
    n_frames = 500
    frames = np.empty((n_frames, H, W), dtype=float)
    for f in range(n_frames):
        k = int(rng.integers(2, 5))
        clean = np.full((H, W), background, dtype=float)
        for idx in rng.choice(len(emitters), size=k, replace=False):
            x0, y0 = emitters[idx]
            clean += gaussian2d(xx, yy, x0, y0, sigma_psf,
                                amp=rng.uniform(16.0, 26.0))
        frames[f] = rng.poisson(clean)
    return frames, xx, yy


def reconstruct_v1(frames, xx, yy):
    """ClassRoomSTORM V1.1 centroid reconstruction."""
    H, W = frames.shape[1:]
    raw_mean = frames.mean(axis=0)
    locs = []
    for fr in frames:
        mask = fr >= np.quantile(fr, THRESHOLD_Q)
        labels, n_comp = label(mask)
        for ci in range(1, n_comp + 1):
            w = fr * (labels == ci)
            s = w.sum()
            if s > 0:
                locs.append(((xx * w).sum() / s, (yy * w).sum() / s))
    recon = np.zeros((H, W), dtype=float)
    for cx, cy in locs:
        recon += gaussian2d(xx, yy, cx, cy, RENDER_SIGMA_PX)
    return raw_mean, recon


def fig_T11():
    frames, xx, yy = simulate_blinking_video()
    raw_mean, recon = reconstruct_v1(frames, xx, yy)
    H, W = raw_mean.shape
    x_cut = W // 2

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=figsize(1.00, 2.7),
                                     sharey=True)
    a1.imshow(raw_mean, origin="lower", extent=(0, W, 0, H), cmap="gray",
              aspect="auto")
    a1.set_title("raw mean image")
    a2.imshow(recon, origin="lower", extent=(0, W, 0, H), cmap="viridis",
              aspect="auto")
    a2.set_title("reconstruction")
    for ax, lab in ((a1, "a"), (a2, "b")):
        ax.axvline(x_cut, color=COL_ACCENT, lw=1.0, ls=(0, (4, 2)))
        ax.set_xlabel(r"$x$ (pixels)")
        style.panel_label(ax, lab)
    a1.set_ylabel(r"$y$ (pixels)")

    raw_p = raw_mean[:, x_cut].astype(float)
    raw_p = raw_p - raw_p.min()
    raw_p = raw_p / raw_p.max() if raw_p.max() > 0 else raw_p
    rec_p = recon[:, x_cut].astype(float)
    rec_p = rec_p / rec_p.max() if rec_p.max() > 0 else rec_p
    y_axis = np.arange(H)
    a3.plot(raw_p, y_axis, color=COL_RAW, label="raw")
    a3.plot(rec_p, y_axis, color=COL_RECON, label="reconstruction")
    a3.set_xlabel("normalized intensity")
    a3.set_title("linked profile (dashed line)")
    a3.set_xlim(-0.04, 1.10)
    a3.set_ylim(0, H)
    a3.legend(loc="upper right", fontsize=7.5)
    style.panel_label(a3, "c")

    fig.tight_layout(w_pad=1.2)
    style.save(fig, OUT, "T11_raw_vs_reconstruction")


def main():
    style.setup()
    print("Generating result figure T11 into", os.path.normpath(OUT))
    fig_T11()
    print("Result figure done.")


if __name__ == "__main__":
    main()
