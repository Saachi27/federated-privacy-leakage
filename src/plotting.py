"""Shared chart style, so every figure in the report reads as one system."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order, checked for colorblind separation on white.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# Distinct markers keep series apart when the report prints in grayscale.
MARKERS = ["o", "s", "^", "D"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "text.color": INK, "axes.titlesize": 11,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 1.8,
        "lines.markersize": 5, "legend.frameon": False,
    })
