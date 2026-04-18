"""Shared plotting utilities for generating charts from experiment results."""

import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str) -> list[dict]:
    """Load experiment results from CSV."""
    with open(filepath, "r") as f:
        return list(csv.DictReader(f))


def filter_results(results: list[dict], **filters) -> list[dict]:
    """Filter results by column values, e.g. filter_results(data, problem='scheduling')."""
    out = results
    for key, val in filters.items():
        out = [r for r in out if r[key] == str(val)]
    return out


def _aggregate_by(results: list[dict], group_key: str, value_key: str):
    """Group results by group_key and compute mean/std of value_key."""
    groups = defaultdict(list)
    for r in results:
        groups[r[group_key]].append(float(r[value_key]))
    keys = sorted(groups.keys(), key=lambda x: (float(x) if x.replace('.', '').isdigit() else x))
    means = [np.mean(groups[k]) for k in keys]
    stds = [np.std(groups[k]) for k in keys]
    return keys, means, stds


def plot_optimality_gap(results: list[dict], problem: str, family: str,
                        save_path: str | None = None):
    """
    Bar chart of mean optimality gap by method for a given problem+family.
    """
    data = filter_results(results, problem=problem, family=family)
    methods = sorted(set(r["method"] for r in data))

    means, stds = [], []
    for m in methods:
        vals = [float(r["optimality_gap"]) for r in data if r["method"] == m]
        means.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals) if vals else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"DP": "#2ecc71", "Greedy_EF": "#e74c3c", "Greedy_HW": "#e67e22",
              "Greedy_BR": "#f39c12", "ML_RF": "#3498db", "ML_MLP": "#9b59b6"}
    bar_colors = [colors.get(m, "#95a5a6") for m in methods]
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=bar_colors, edgecolor="black")
    ax.set_ylabel("Optimality Gap (lower is better)")
    ax.set_title(f"Optimality Gap — {problem} ({family})")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_runtime_scaling(results: list[dict], problem: str,
                         save_path: str | None = None):
    """
    Line chart of mean runtime vs. instance size (n) for each method.
    """
    data = filter_results(results, problem=problem)
    methods = sorted(set(r["method"] for r in data))

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"DP": "o", "Greedy_EF": "s", "Greedy_HW": "^",
               "Greedy_BR": "D", "ML_RF": "v", "ML_MLP": "P"}
    colors = {"DP": "#2ecc71", "Greedy_EF": "#e74c3c", "Greedy_HW": "#e67e22",
              "Greedy_BR": "#f39c12", "ML_RF": "#3498db", "ML_MLP": "#9b59b6"}

    for m in methods:
        method_data = [r for r in data if r["method"] == m]
        keys, means, stds = _aggregate_by(method_data, "n", "runtime_seconds")
        x = [int(k) for k in keys]
        ax.errorbar(x, means, yerr=stds, label=m, marker=markers.get(m, "o"),
                    color=colors.get(m, "#95a5a6"), capsize=3)

    ax.set_xlabel("Instance Size (n)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(f"Runtime Scaling — {problem}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_quality_vs_size(results: list[dict], problem: str,
                         save_path: str | None = None):
    """
    Line chart of mean objective value vs. instance size for each method.
    """
    data = filter_results(results, problem=problem)
    methods = sorted(set(r["method"] for r in data))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"DP": "#2ecc71", "Greedy_EF": "#e74c3c", "Greedy_HW": "#e67e22",
              "Greedy_BR": "#f39c12", "ML_RF": "#3498db", "ML_MLP": "#9b59b6"}

    for m in methods:
        method_data = [r for r in data if r["method"] == m]
        keys, means, stds = _aggregate_by(method_data, "n", "objective_value")
        x = [int(k) for k in keys]
        ax.errorbar(x, means, yerr=stds, label=m,
                    color=colors.get(m, "#95a5a6"), capsize=3)

    ax.set_xlabel("Instance Size (n)")
    ax.set_ylabel("Objective Value")
    ax.set_title(f"Solution Quality — {problem}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig
