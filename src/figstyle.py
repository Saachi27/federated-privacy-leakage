"""One look for every journal figure (scripts/make_journal_figures.py)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MULTI = "#2a78d6"
SINGLE = "#eb6834"
CONTROL = "#1baf7a"
BASE = "#8a8984"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# Fixed order for the four single-step settings (slots 1-4).
SETTING_COLORS = {
    "mnist-fresh": "#2a78d6",
    "mnist-trained": "#eb6834",
    "cifar10-fresh": "#1baf7a",
    "mnist-fedavg": "#eda100",
}
SETTING_MARKERS = {"mnist-fresh": "o", "mnist-trained": "s", "cifar10-fresh": "^", "mnist-fedavg": "D"}
SETTING_NAMES = {
    "mnist-fresh": "MNIST fresh",
    "mnist-trained": "MNIST trained",
    "cifar10-fresh": "CIFAR-10 fresh",
    "mnist-fedavg": "MNIST FedAvg update",
}

# Journal page widths (Elsevier: single column ~90 mm, double ~190 mm).
SINGLE_COL = 3.54
DOUBLE_COL = 7.48


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, out_dir, name):
    """Write name.pdf (for the paper) and name.png (for review); return the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    print("saved", " ".join(paths))
    return paths
