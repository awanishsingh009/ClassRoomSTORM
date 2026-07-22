"""
Generate the diagram figures for ClassRoomSTORM V2:

  T10 - reconstruction pipeline (compact two-row flowchart)
  H1  - concept loop
  H9  - output-file map

These are drawn with Matplotlib (not Graphviz) so they share the exact
Computer-Modern / LaTeX typography of every other figure and document text,
and so they can be sized precisely for the final PDF. Each figure is drawn in
inch-coordinates and inserted at full text width (scale 1.0).

Output: T10 -> ../figures/theory/ ;  H1, H9 -> ../figures/help/
Run:    python make_diagrams.py
"""
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import COL_AXIS

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.join(HERE, "..", "figures", "theory")
HELP = os.path.join(HERE, "..", "figures", "help")

BOX_FILL = "#eef2f7"
BOX_EDGE = "#3a4a5c"
GROUP_VE = "#e7f3e8"
GROUP_RC = "#e7eef7"


def _canvas(width_in, height_in):
    """A figure with one inch-coordinate axes filling it (1 unit = 1 inch)."""
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_in)
    ax.set_ylim(0, height_in)
    ax.axis("off")
    return fig, ax


def _box(ax, cx, cy, w, h, fc=BOX_FILL, lw=0.9):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.07",
        facecolor=fc, edgecolor=BOX_EDGE, linewidth=lw))


def _arrow(ax, p0, p1):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=COL_AXIS, lw=1.1,
                                shrinkA=1.5, shrinkB=1.5))


# ---------------------------------------------------------------- T10
def fig_T10():
    """Compact two-row reconstruction pipeline (full text width)."""
    W, H = 6.38, 3.55
    fig, ax = _canvas(W, H)
    bw, bh = 1.46, 1.06
    xs = [0.85, 2.43, 4.01, 5.59]            # four columns
    y_top, y_bot = 2.62, 0.92
    steps = [
        (xs[0], y_top, "1. Load blinking\nvideo"),
        (xs[1], y_top, "2. Threshold frame\n(top 0.5\\%)"),
        (xs[2], y_top, "3. Intensity-weighted\ncentroid"),
        (xs[3], y_top, "4. Accumulate\nlocalizations"),
        (xs[3], y_bot, "5. Gaussian render\n($\\sigma \\approx 1.5$ px)"),
        (xs[2], y_bot, "6. Raw mean\nimage"),
        (xs[1], y_bot, "7. Compare raw\nvs.\\ reconstruction"),
    ]
    for cx, cy, text in steps:
        _box(ax, cx, cy, bw, bh)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=8.5,
                color=COL_AXIS, linespacing=1.35)
    # top row, left to right
    for i in range(3):
        _arrow(ax, (xs[i] + bw / 2, y_top), (xs[i + 1] - bw / 2, y_top))
    # connector down to the second row
    _arrow(ax, (xs[3], y_top - bh / 2), (xs[3], y_bot + bh / 2))
    # bottom row, right to left
    _arrow(ax, (xs[3] - bw / 2, y_bot), (xs[2] + bw / 2, y_bot))
    _arrow(ax, (xs[2] - bw / 2, y_bot), (xs[1] + bw / 2, y_bot))
    style.save(fig, THEORY, "T10_pipeline")


# ---------------------------------------------------------------- H1
def fig_H1():
    """The concept loop: blink, localize, accumulate, compare (full width)."""
    W, H = 6.38, 1.72
    fig, ax = _canvas(W, H)
    bw, bh = 1.40, 1.16
    xs = [0.84, 2.41, 3.98, 5.55]
    cy = 0.96
    steps = [("Blink", "a few emitters\non per frame"),
             ("Localize", "find the centre\nof each spot"),
             ("Accumulate", "collect centres\nfrom all frames"),
             ("Compare", "raw mean vs.\nreconstruction")]
    for x, (title, sub) in zip(xs, steps):
        _box(ax, x, cy, bw, bh)
        ax.text(x, cy + 0.30, r"\textbf{%s}" % title if style.using_tex()
                else title, ha="center", va="center", fontsize=9.5,
                color=COL_AXIS)
        ax.text(x, cy - 0.16, sub, ha="center", va="center", fontsize=7.6,
                color=COL_AXIS, linespacing=1.3)
    for i in range(3):
        _arrow(ax, (xs[i] + bw / 2, cy), (xs[i + 1] - bw / 2, cy))
    style.save(fig, HELP, "H1_concept_loop")


# ---------------------------------------------------------------- H9
def fig_H9():
    """Map of the files written by each workflow (two labelled columns)."""
    W, H = 6.38, 3.35
    fig, ax = _canvas(W, H)

    def column(x0, x1, accent, fill, header, files):
        y0, y1 = 0.18, 3.06
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.0,rounding_size=0.10",
            facecolor=fill, edgecolor=accent, linewidth=1.2))
        hdr = r"\textbf{%s}" % header if style.using_tex() else header
        ax.text((x0 + x1) / 2, y1 - 0.26, hdr, ha="center", va="center",
                fontsize=9.2, color=accent,
                fontweight=("normal" if style.using_tex() else "bold"))
        ax.plot([x0 + 0.18, x1 - 0.18], [y1 - 0.50, y1 - 0.50],
                color=accent, lw=0.6, alpha=0.5)
        top = y1 - 0.84
        step = (top - y0 - 0.22) / max(1, len(files) - 1) if len(files) > 1 else 0
        for i, fname in enumerate(files):
            txt = r"\texttt{%s}" % fname if style.using_tex() else fname
            ax.text(x0 + 0.26, top - i * step, txt, ha="left", va="center",
                    fontsize=8.3, color=COL_AXIS)

    ve = ["blinking\\_video.mp4", "truth.csv", "metadata.json",
          "preview\\_frame.png", "pattern\\_mask.png"]
    rc = ["raw\\_mean.npy / .png", "superres.npy / .png",
          "raw\\_vs\\_superres.png", "localizations.csv", "summary.json",
          "plot\\_*.png  (diagnostics)", "pipeline\\_report.html / .md"]
    column(0.12, 3.07, "#2e7d32", GROUP_VE, "Virtual Experiment writes", ve)
    column(3.31, 6.26, "#1f5f9e", GROUP_RC, "SR Reconstruction writes", rc)
    style.save(fig, HELP, "H9_output_map")


def main():
    style.setup()
    print("Generating diagram figures (T10, H1, H9)")
    fig_T10()
    fig_H1()
    fig_H9()
    print("Diagram figures done.")


if __name__ == "__main__":
    main()
