"""Shared benchmarking utilities for timing solvers and logging results."""

import time
import csv
import os
from typing import Callable, Any


def time_solver(solver_fn: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Run a solver and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = solver_fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def compute_optimality_gap(dp_value: float, method_value: float) -> float:
    """Compute (dp_value - method_value) / dp_value. Returns 0 if dp_value is 0."""
    if dp_value == 0:
        return 0.0
    return (dp_value - method_value) / dp_value


RESULTS_HEADER = [
    "problem", "method", "family", "instance_id", "n",
    "objective_value", "optimality_gap", "runtime_seconds"
]


def log_result(filepath: str, row: dict):
    """Append a single result row to a CSV file, creating it with headers if needed."""
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(problem: str, family: str, instance_id: int, n: int,
                   instance_args: dict,
                   solvers: dict[str, Callable],
                   dp_value: float,
                   results_path: str):
    """
    Run all solvers on one instance and log results.

    solvers: dict mapping method name -> callable that returns objective value
    dp_value: the DP-optimal value for this instance (for gap computation)
    """
    for method_name, solver_fn in solvers.items():
        value, elapsed = time_solver(solver_fn)
        gap = compute_optimality_gap(dp_value, value)
        log_result(results_path, {
            "problem": problem,
            "method": method_name,
            "family": family,
            "instance_id": instance_id,
            "n": n,
            "objective_value": value,
            "optimality_gap": gap,
            "runtime_seconds": elapsed,
        })
