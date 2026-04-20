"""
Random knapsack instance generator.

Supports two distribution families:
  - Uncorrelated: weights and values drawn independently from uniform distributions
  - Correlated: values positively correlated with weights (v_i = w_i + noise)
"""

import numpy as np


def generate_instance(n, W, distribution="uncorrelated", rng=None):
    """Generate a single 0/1 Knapsack instance.

    Args:
        n: int, number of items
        W: int, knapsack capacity
        distribution: "uncorrelated" or "correlated"
        rng: numpy RandomState (for reproducibility)

    Returns:
        dict with keys: weights, values, capacity, n, distribution
    """
    if rng is None:
        rng = np.random.RandomState()

    if distribution == "uncorrelated":
        weights = rng.randint(1, W // 2 + 1, size=n)
        values = rng.randint(1, W // 2 + 1, size=n)
    elif distribution == "correlated":
        weights = rng.randint(1, W // 2 + 1, size=n)
        noise = rng.randint(-W // 10, W // 10 + 1, size=n)
        values = np.clip(weights + noise, 1, None)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return {
        "weights": weights.tolist(),
        "values": values.tolist(),
        "capacity": W,
        "n": n,
        "distribution": distribution,
    }


def generate_dataset(num_instances, n, W, distribution="uncorrelated", seed=42):
    """Generate multiple instances for training/testing.

    Args:
        num_instances: how many instances to generate
        n: items per instance
        W: capacity per instance
        distribution: "uncorrelated" or "correlated"
        seed: random seed

    Returns:
        list of instance dicts
    """
    rng = np.random.RandomState(seed)
    return [generate_instance(n, W, distribution, rng) for _ in range(num_instances)]
