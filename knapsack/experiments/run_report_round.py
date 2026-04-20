"""
Report-round experiment runner.

Expanded experiments for the IEEE report:
- 3 size tiers: n=20/W=50, n=50/W=100, n=100/W=200
- 20 instances per family per tier
- 3 test families: uncorrelated, correlated, adversarial
- 4 solvers: DP, Greedy, ML (RF), ML (MLP)
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.test_families import generate_family
from solvers.dp_solver import dp_solve
from solvers.greedy_solver import greedy_solve
from solvers.ml_solver import ml_solve
from config import REPORT_SIZE_TIERS, REPORT_INSTANCES_PER_FAMILY, RESULTS_DIR
from utils import save_results_csv


FAMILIES = ["uncorrelated", "correlated", "adversarial"]
SOLVERS = {
    "dp": lambda w, v, c: dp_solve(w, v, c),
    "greedy": lambda w, v, c: greedy_solve(w, v, c),
    "ml_rf": lambda w, v, c: ml_solve(w, v, c, model_name="rf"),
    "ml_mlp": lambda w, v, c: ml_solve(w, v, c, model_name="mlp"),
}


def run_experiment(instance, solver_fn):
    """Run a single solver on a single instance, return metrics."""
    w, v, c = instance["weights"], instance["values"], instance["capacity"]
    start = time.perf_counter()
    value, selected = solver_fn(w, v, c)
    elapsed = time.perf_counter() - start
    total_weight = sum(wi * si for wi, si in zip(w, selected))
    return {
        "value": value,
        "weight": total_weight,
        "time_s": elapsed,
        "feasible": total_weight <= c,
        "items_selected": sum(selected),
    }


def run_all():
    """Run the full report-round experiment."""
    print("=" * 70)
    print("REPORT-ROUND EXPERIMENTS")
    print("=" * 70)

    all_rows = []
    family_seeds = {"uncorrelated": 1000, "correlated": 2000, "adversarial": 3000}

    for tier in REPORT_SIZE_TIERS:
        n, W = tier["n"], tier["W"]
        print(f"\n--- Size tier: n={n}, W={W} ---")

        for family in FAMILIES:
            instances = generate_family(
                family, n, W, REPORT_INSTANCES_PER_FAMILY,
                seed=family_seeds[family]
            )
            print(f"\n  Family: {family} ({len(instances)} instances)")

            for inst_idx, inst in enumerate(instances):
                dp_result = run_experiment(inst, SOLVERS["dp"])
                opt_val = dp_result["value"]

                for solver_name, solver_fn in SOLVERS.items():
                    if solver_name == "dp":
                        result = dp_result
                    else:
                        result = run_experiment(inst, solver_fn)

                    gap = (opt_val - result["value"]) / opt_val * 100 if opt_val > 0 else 0.0

                    row = {
                        "n": n,
                        "W": W,
                        "family": family,
                        "instance_idx": inst_idx,
                        "solver": solver_name,
                        "value": result["value"],
                        "optimal_value": opt_val,
                        "opt_gap_pct": round(gap, 2),
                        "weight": result["weight"],
                        "capacity": W,
                        "feasible": result["feasible"],
                        "time_s": round(result["time_s"], 6),
                        "items_selected": result["items_selected"],
                    }
                    all_rows.append(row)

            # Print summary
            for solver_name in SOLVERS:
                solver_rows = [r for r in all_rows
                              if r["n"] == n and r["family"] == family and r["solver"] == solver_name]
                avg_gap = sum(r["opt_gap_pct"] for r in solver_rows) / len(solver_rows)
                avg_time = sum(r["time_s"] for r in solver_rows) / len(solver_rows)
                max_gap = max(r["opt_gap_pct"] for r in solver_rows)
                print(f"    {solver_name:<10} avg_gap={avg_gap:6.2f}%  max_gap={max_gap:6.2f}%  avg_time={avg_time:.6f}s")

    # Save to CSV
    fieldnames = [
        "n", "W", "family", "instance_idx", "solver",
        "value", "optimal_value", "opt_gap_pct", "weight", "capacity",
        "feasible", "time_s", "items_selected",
    ]
    csv_path = os.path.join(RESULTS_DIR, "report_round_results.csv")
    save_results_csv(csv_path, all_rows, fieldnames)
    print(f"\nResults saved to: {csv_path}")
    print(f"Total rows: {len(all_rows)}")

    return all_rows


if __name__ == "__main__":
    run_all()
