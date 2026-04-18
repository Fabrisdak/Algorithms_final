"""
Instance generators for Weighted Interval Scheduling.

Three families per the experiment plan:
  - Random Family 1 (sparse): intervals with low overlap probability.
  - Random Family 2 (dense): intervals with high overlap probability.
  - Adversarial: hand-crafted cases where greedy provably fails.
"""

import random
import json
import os


def generate_sparse_random(n: int, time_horizon: int = None,
                           max_duration: int = None,
                           weight_range: tuple[int, int] = (1, 100),
                           seed: int | None = None) -> list[dict]:
    """
    Random Family 1 — sparse overlap.
    Intervals are spread across a wide time horizon relative to their duration.
    """
    if time_horizon is None:
        time_horizon = n * 10
    if max_duration is None:
        max_duration = max(time_horizon // n, 2)

    rng = random.Random(seed)
    intervals = []
    for _ in range(n):
        duration = rng.randint(1, max_duration)
        start = rng.randint(0, time_horizon - duration)
        weight = rng.randint(*weight_range)
        intervals.append({"start": start, "finish": start + duration, "weight": weight})
    return intervals


def generate_dense_random(n: int, time_horizon: int = None,
                          max_duration: int = None,
                          weight_range: tuple[int, int] = (1, 100),
                          seed: int | None = None) -> list[dict]:
    """
    Random Family 2 — dense overlap.
    Intervals are packed into a narrow time horizon, forcing heavy overlap.
    """
    if time_horizon is None:
        time_horizon = max(n * 2, 10)
    if max_duration is None:
        max_duration = max(time_horizon // 2, 2)

    rng = random.Random(seed)
    intervals = []
    for _ in range(n):
        duration = rng.randint(1, max_duration)
        start = rng.randint(0, max(time_horizon - duration, 0))
        weight = rng.randint(*weight_range)
        intervals.append({"start": start, "finish": start + duration, "weight": weight})
    return intervals


def generate_adversarial_greedy(n: int = 5) -> list[dict]:
    """
    Hand-crafted case where greedy heuristics fail.

    Construction: one long, very heavy interval spanning [0, 2n], plus n short
    intervals of moderate weight packed inside [0, 2n] that collectively outweigh
    the long one. Earliest-finish greedy picks the short ones (misses weight),
    highest-weight greedy picks the long one (misses the better combination),
    but DP finds the optimal non-overlapping subset.
    """
    span = 2 * n
    big_weight = n * 10
    small_weight = big_weight // n + 2

    intervals = [{"start": 0, "finish": span, "weight": big_weight}]

    slot_width = span // n
    for i in range(n):
        s = i * slot_width
        f = s + slot_width
        intervals.append({"start": s, "finish": f, "weight": small_weight})

    return intervals


def generate_adversarial_ef(n: int = 4) -> list[dict]:
    """
    Case where earliest-finish greedy is specifically bad.
    Short low-weight intervals finish early and block high-weight longer ones.
    """
    intervals = []
    for i in range(n):
        intervals.append({"start": i * 2, "finish": i * 2 + 1, "weight": 1})
    intervals.append({"start": 0, "finish": n * 2, "weight": n * 10})
    return intervals


def save_instance(intervals: list[dict], filepath: str):
    """Save an instance to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(intervals, f, indent=2)


def load_instance(filepath: str) -> list[dict]:
    """Load an instance from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)
