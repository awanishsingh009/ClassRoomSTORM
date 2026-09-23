"""Portable publication-style defaults for generated diagnostics; no TeX needed."""
def configure(plt):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "text.usetex": False,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "grid.linewidth": .5,
        "text.color": "#20252C", "axes.labelcolor": "#20252C",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
