"""
Seaborn/matplotlib setup for consistent, high-contrast, colorblind-friendly charts.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from config import CHART_DPI, CHART_FIGSIZE, SEABORN_PALETTE


def setup_style():
    """Apply project-wide chart styling."""
    sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)
    plt.rcParams.update({
        "figure.figsize": CHART_FIGSIZE,
        "figure.dpi": CHART_DPI,
        "savefig.dpi": CHART_DPI,

        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })


# Solver display names and consistent colors
SOLVER_DISPLAY = {
    "dp": "DP (Exact)",
    "greedy": "Greedy",
    "ml_rf": "ML (RF)",
    "ml_mlp": "ML (MLP)",
}

SOLVER_ORDER = ["dp", "greedy", "ml_rf", "ml_mlp"]

FAMILY_DISPLAY = {
    "uncorrelated": "Uncorrelated",
    "correlated": "Correlated",
    "adversarial": "Adversarial",
}
