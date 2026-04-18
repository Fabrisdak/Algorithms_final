"""
Experiment runner for Weighted Interval Scheduling.

Runs all methods (DP, 3 Greedy heuristics, ML RF, ML MLP) across all instance
families and sizes, logging results to CSV for chart generation.
"""

import os
import sys
import time

# Repo root (parent of `scheduling/` package)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scheduling.dp_solver import solve as dp_solve
from scheduling.greedy_solver import (
    solve_earliest_finish,
    solve_highest_weight,
    solve_best_ratio,
)
from scheduling.ml_solver import (
    SchedulingMLSolver,
    generate_training_data,
)
from scheduling.data_gen import (
    generate_sparse_random,
    generate_dense_random,
    generate_adversarial_greedy,
)
from scheduling.utils.benchmarking import time_solver, compute_optimality_gap, log_result

RESULTS_PATH = os.path.join(_PROJECT_ROOT, "scheduling", "results", "results.csv")


def train_ml_models():
    """Train ML models on DP-labeled data from both sparse and dense distributions."""
    print("Training ML models...")
    solver = SchedulingMLSolver()

    import numpy as np
    X_sparse, y_sparse = generate_training_data(generate_sparse_random, n_instances=300, n_range=(5, 40), seed=42)
    X_dense, y_dense = generate_training_data(generate_dense_random, n_instances=300, n_range=(5, 40), seed=1000)

    X = np.vstack([X_sparse, X_dense])
    y = np.concatenate([y_sparse, y_dense])

    solver.fit(X, y)
    print(f"  Training samples: {len(y)} | Positive rate: {y.mean():.2%}")
    return solver


def run_single_instance(intervals, instance_id, n, family, ml_solver, results_path):
    """Run all methods on a single instance and log results."""
    (dp_value, dp_selected), dp_time = time_solver(dp_solve, intervals)
    (ef_value, _), ef_time = time_solver(solve_earliest_finish, intervals)
    (hw_value, _), hw_time = time_solver(solve_highest_weight, intervals)
    (br_value, _), br_time = time_solver(solve_best_ratio, intervals)
    (rf_value, _), rf_time = time_solver(ml_solver.predict_rf, intervals)
    (mlp_value, _), mlp_time = time_solver(ml_solver.predict_mlp, intervals)

    methods = {
        "DP": (dp_value, dp_time),
        "Greedy_EF": (ef_value, ef_time),
        "Greedy_HW": (hw_value, hw_time),
        "Greedy_BR": (br_value, br_time),
        "ML_RF": (rf_value, rf_time),
        "ML_MLP": (mlp_value, mlp_time),
    }

    for method_name, (value, elapsed) in methods.items():
        gap = compute_optimality_gap(dp_value, value)
        log_result(results_path, {
            "problem": "scheduling",
            "method": method_name,
            "family": family,
            "instance_id": instance_id,
            "n": n,
            "objective_value": value,
            "optimality_gap": gap,
            "runtime_seconds": elapsed,
        })


def run_slides_round(ml_solver):
    """
    Slides-round experiments: 2 size tiers x ~10 instances per family + failure cases.
    """
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)

    size_tiers = [20, 50]
    n_trials = 10

    families = {
        "sparse": generate_sparse_random,
        "dense": generate_dense_random,
    }

    instance_id = 0
    for family_name, gen_fn in families.items():
        for n in size_tiers:
            print(f"  Running {family_name} family, n={n}...")
            for trial in range(n_trials):
                intervals = gen_fn(n, seed=trial * 100 + n)
                run_single_instance(intervals, instance_id, n, family_name, ml_solver, RESULTS_PATH)
                instance_id += 1

    print("  Running adversarial cases...")
    for n in [5, 8, 12]:
        intervals = generate_adversarial_greedy(n)
        run_single_instance(intervals, instance_id, len(intervals), "adversarial", ml_solver, RESULTS_PATH)
        instance_id += 1

    print(f"Slides-round complete. {instance_id} instances logged to {RESULTS_PATH}")


def run_report_round(ml_solver):
    """
    Report-round experiments: 3 size tiers x ~15-20 instances per family + more failure cases.
    Appends to existing results.
    """
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    size_tiers = [20, 50, 100]
    n_trials = 15

    families = {
        "sparse": generate_sparse_random,
        "dense": generate_dense_random,
    }

    instance_id = 1000
    for family_name, gen_fn in families.items():
        for n in size_tiers:
            print(f"  Running {family_name} family, n={n}...")
            for trial in range(n_trials):
                intervals = gen_fn(n, seed=2000 + trial * 100 + n)
                run_single_instance(intervals, instance_id, n, family_name, ml_solver, RESULTS_PATH)
                instance_id += 1

    print("  Running expanded adversarial cases...")
    for n in [5, 8, 12, 20, 30]:
        intervals = generate_adversarial_greedy(n)
        run_single_instance(intervals, instance_id, len(intervals), "adversarial", ml_solver, RESULTS_PATH)
        instance_id += 1

    print(f"Report-round complete. Additional instances logged.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", choices=["slides", "report", "both"], default="slides")
    args = parser.parse_args()

    ml_solver = train_ml_models()

    if args.round in ("slides", "both"):
        print("\n=== SLIDES ROUND ===")
        run_slides_round(ml_solver)

    if args.round in ("report", "both"):
        print("\n=== REPORT ROUND ===")
        run_report_round(ml_solver)

    print("\nDone!")
