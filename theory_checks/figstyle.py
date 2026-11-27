"""Shared matplotlib style for the figures made by the scripts in this folder.

Matches the original figures of the paper (plot_qubit_vs_metrics.py, misc_plots.py, comparison_to_qaoa.py):
matplotlib's built-in defaults (DejaVu Sans, full axes box, framed legends) with large fonts, independent of any
system matplotlibrc.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"


def apply(size=10):
    matplotlib.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": size, "axes.titlesize": size, "axes.labelsize": size,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1, "legend.fontsize": size - 1,
        "lines.linewidth": 1.8, "lines.markersize": 5, "savefig.dpi": 300, "savefig.bbox": "tight",
        "legend.framealpha": 0.9,
    })
    return plt
