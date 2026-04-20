"""
Generate report-round charts from expanded experiment results.

Charts produced:
  1. Optimality gap box plots (by family + solver, all 3 size tiers)
  2. Runtime scaling line plot (n vs runtime, one line per solver)
  3. Scalability: gap vs problem size
  4. Mean-gap heatmap by solver, family, and size
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from charts.plot_utils import setup_style, SOLVER_DISPLAY, SOLVER_ORDER, FAMILY_DISPLAY
from config import RESULTS_DIR, CHARTS_DIR


def load_results():
    csv_path = os.path.join(RESULTS_DIR, "report_round_results.csv")
    df = pd.read_csv(csv_path)
    df["solver_label"] = df["solver"].map(SOLVER_DISPLAY)
    df["family_label"] = df["family"].map(FAMILY_DISPLAY)
    df["size_label"] = df.apply(lambda r: f"n={int(r['n'])}", axis=1)
    return df


def chart_gap_boxplots(df):
    """Box plots of optimality gap by family, one panel per size tier."""
    setup_style()

    plot_df = df[df["solver"] != "dp"].copy()
    size_tiers = sorted(plot_df["n"].unique())
    family_order = [FAMILY_DISPLAY[f] for f in ["uncorrelated", "correlated", "adversarial"]]
    solver_order = [SOLVER_DISPLAY[s] for s in SOLVER_ORDER if s != "dp"]

    boxplot_fonts = {
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
    }

    with plt.rc_context(boxplot_fonts):
        fig, axes = plt.subplots(3, 1, figsize=(3.58, 4.85), sharex=True, sharey=True)
        legend_handles = None
        legend_labels = None

        for ax, n_val in zip(axes, size_tiers):
            tier_df = plot_df[plot_df["n"] == n_val]
            W_val = tier_df["W"].iloc[0]

            sns.boxplot(
                data=tier_df,
                x="family_label",
                y="opt_gap_pct",
                hue="solver_label",
                order=family_order,
                hue_order=solver_order,
                ax=ax,
                width=0.72,
                linewidth=0.8,
                fliersize=2,
                saturation=0.9,
            )
            ax.set_title(f"n={n_val}, W={W_val}", pad=5)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_ylim(0, 98)
            ax.grid(True, axis="y", alpha=0.3)
            ax.grid(False, axis="x")

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            if ax.legend_ is not None:
                ax.legend_.remove()

        axes[-1].set_xlabel("Test Family", labelpad=4)
        for ax in axes:
            ax.tick_params(axis="x", rotation=0, length=0)

        fig.text(0.018, 0.49, "Optimality Gap (%)", rotation=90, va="center", ha="center", fontsize=7.8)
        fig.text(
            0.5,
            0.995,
            "Optimality Gap Distribution",
            ha="center",
            va="top",
            fontsize=8.4,
            fontweight="semibold",
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.56, 0.955),
            ncol=3,
            frameon=False,
            handlelength=1.05,
            columnspacing=0.55,
            borderaxespad=0,
        )
        sns.despine(fig=fig)
        fig.subplots_adjust(left=0.155, right=0.985, top=0.85, bottom=0.105, hspace=0.42)

        path = os.path.join(CHARTS_DIR, "report_gap_boxplots.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def chart_runtime_scaling(df):
    """Line plot: runtime vs problem size, one line per solver."""
    setup_style()

    # Average runtime across all families for each (solver, n)
    avg_df = df.groupby(["solver", "solver_label", "n"])["time_s"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    for solver in SOLVER_ORDER:
        s_df = avg_df[avg_df["solver"] == solver].sort_values("n")
        label = SOLVER_DISPLAY[solver]
        ax.plot(s_df["n"], s_df["time_s"], marker="o", linewidth=2, markersize=8, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Number of Items (n)")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_title("Runtime Scaling with Problem Size")
    ax.set_xticks(sorted(df["n"].unique()))
    ax.legend(title="Solver")
    ax.grid(True, alpha=0.3)

    path = os.path.join(CHARTS_DIR, "report_runtime_scaling.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def chart_gap_vs_size(df):
    """Line plot: mean gap vs problem size by solver, across families."""
    setup_style()

    plot_df = df[df["solver"] != "dp"].copy()
    families = ["uncorrelated", "correlated", "adversarial"]
    size_order = sorted(df["n"].unique())
    solvers = [s for s in SOLVER_ORDER if s != "dp"]

    gap_scaling_fonts = {
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
    }

    with plt.rc_context(gap_scaling_fonts):
        fig, axes = plt.subplots(3, 1, figsize=(3.58, 4.55), sharex=True, sharey=False)
        legend_handles = []
        legend_labels = []

        for i, (ax, family) in enumerate(zip(axes, families)):
            fam_df = plot_df[plot_df["family"] == family]
            avg = (
                fam_df.groupby(["solver", "solver_label", "n"])["opt_gap_pct"]
                .agg(["mean", "std"])
                .reset_index()
            )
            avg["std"] = avg["std"].fillna(0)

            for solver in solvers:
                s_df = avg[avg["solver"] == solver].sort_values("n")
                label = SOLVER_DISPLAY[solver]
                mean = s_df["mean"].to_numpy()
                std = s_df["std"].to_numpy()
                lower_err = np.minimum(std, mean)
                yerr = np.vstack([lower_err, std])
                artist = ax.errorbar(
                    s_df["n"],
                    mean,
                    yerr=yerr,
                    marker="o",
                    linewidth=1.3,
                    markersize=3.4,
                    capsize=2.5,
                    capthick=1,
                    elinewidth=1,
                    label=label,
                )
                if i == 0:
                    legend_handles.append(artist)
                    legend_labels.append(label)

            ax.set_title(FAMILY_DISPLAY[family], pad=5)
            ax.set_xticks(size_order)
            ax.set_ylim(0, max((avg["mean"] + avg["std"]).max() * 1.08, 1))
            ax.grid(True, axis="y", alpha=0.3)
            ax.grid(True, axis="x", alpha=0.12)

        axes[-1].set_xlabel("Number of Items (n)", labelpad=4)
        fig.text(0.018, 0.49, "Mean Gap (%)", rotation=90, va="center", ha="center", fontsize=7.8)
        fig.text(
            0.5,
            0.995,
            "Optimality Gap Scaling",
            ha="center",
            va="top",
            fontsize=8.4,
            fontweight="semibold",
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.56, 0.955),
            ncol=3,
            frameon=False,
            handlelength=1.05,
            columnspacing=0.55,
            borderaxespad=0,
        )
        sns.despine(fig=fig)
        fig.subplots_adjust(left=0.155, right=0.985, top=0.85, bottom=0.105, hspace=0.4)

        path = os.path.join(CHARTS_DIR, "report_gap_scaling.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def chart_combined_heatmap(df):
    """Heatmap: mean gap by (family × size) for each non-DP solver."""
    setup_style()

    plot_df = df[df["solver"] != "dp"].copy()
    solvers = [s for s in SOLVER_ORDER if s != "dp"]
    family_order = [FAMILY_DISPLAY[f] for f in ["uncorrelated", "correlated", "adversarial"]]
    size_order = sorted(plot_df["n"].unique())
    vmax = plot_df.groupby(["solver", "family_label", "n"])["opt_gap_pct"].mean().max()

    heatmap_fonts = {
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    }

    with plt.rc_context(heatmap_fonts):
        fig = plt.figure(figsize=(3.5, 4.95))
        gs = fig.add_gridspec(
            nrows=3,
            ncols=2,
            width_ratios=[1, 0.055],
            hspace=0.4,
            wspace=0.08,
        )
        axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        cbar_ax = fig.add_subplot(gs[:, 1])

        for i, (ax, solver) in enumerate(zip(axes, solvers)):
            s_df = plot_df[plot_df["solver"] == solver]
            pivot = s_df.pivot_table(
                values="opt_gap_pct", index="family_label", columns="n", aggfunc="mean"
            )
            pivot = pivot.reindex(index=family_order, columns=size_order)

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".1f",
                annot_kws={"fontsize": 7.3},
                cmap="YlOrRd",
                ax=ax,
                cbar=(i == 0),
                cbar_ax=cbar_ax if i == 0 else None,
                cbar_kws={"label": "Gap %"},
                vmin=0,
                vmax=vmax,
                linewidths=0.5,
                linecolor="white",
            )
            ax.set_title(SOLVER_DISPLAY[solver], pad=5)
            ax.set_xlabel("n (items)" if i == len(axes) - 1 else "")
            ax.set_ylabel("Test Family" if i == 1 else "")
            ax.tick_params(axis="x", rotation=0, length=0)
            ax.tick_params(axis="y", rotation=0, length=0)

        cbar_ax.tick_params(labelsize=7.5)
        cbar_ax.set_ylabel("Gap %", fontsize=8, labelpad=5)
        fig.suptitle("Mean Optimality Gap (%)", fontsize=9.5, y=0.985)
        fig.subplots_adjust(left=0.29, right=0.93, top=0.92, bottom=0.085)

        path = os.path.join(CHARTS_DIR, "report_gap_heatmap.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def main():
    print("Generating report-round charts...\n")
    df = load_results()

    chart_gap_boxplots(df)
    chart_runtime_scaling(df)
    chart_gap_vs_size(df)
    chart_combined_heatmap(df)

    print("\nAll report-round charts generated.")


if __name__ == "__main__":
    main()
