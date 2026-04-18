import os
import shutil
import sys

# Repo root (parent of `scheduling/` package)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scheduling.utils.plotting import (
    load_results,
    plot_optimality_gap,
    plot_runtime_scaling,
    plot_quality_vs_size,
)

RESULTS_PATH = os.path.join(_PROJECT_ROOT, "scheduling", "results", "results.csv")
# One-time migration if CSV still lives at old paths (checked in order)
_LEGACY_REPO_RESULTS = os.path.join(_PROJECT_ROOT, "results", "scheduling", "results.csv")
_LEGACY_CHARTS_CSV = os.path.join(_PROJECT_ROOT, "scheduling", "charts", "results.csv")
_LEGACY_SCHED_ROOT_CSV = os.path.join(_PROJECT_ROOT, "scheduling", "results.csv")
# PNGs next to report .tex files (matches \graphicspath{{report/}} in main.tex)
CHARTS_DIR = os.path.join(_PROJECT_ROOT, "report")


def main():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    if not os.path.isfile(RESULTS_PATH):
        for legacy in (
            _LEGACY_REPO_RESULTS,
            _LEGACY_CHARTS_CSV,
            _LEGACY_SCHED_ROOT_CSV,
        ):
            if os.path.isfile(legacy):
                shutil.copy2(legacy, RESULTS_PATH)
                print(f"Migrated results from {legacy}")
                break
    results = load_results(RESULTS_PATH)
    print(f"Loaded {len(results)} result rows")

    for family in ["sparse", "dense", "adversarial"]:
        print(f"  Generating optimality gap chart for {family}...")
        plot_optimality_gap(
            results, "scheduling", family,
            save_path=os.path.join(CHARTS_DIR, f"opt_gap_{family}.png")
        )

    print("  Generating runtime scaling chart...")
    plot_runtime_scaling(
        results, "scheduling",
        save_path=os.path.join(CHARTS_DIR, "runtime_scaling.png")
    )

    print("  Generating solution quality chart...")
    plot_quality_vs_size(
        results, "scheduling",
        save_path=os.path.join(CHARTS_DIR, "solution_quality.png")
    )

    print(f"Charts saved to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
