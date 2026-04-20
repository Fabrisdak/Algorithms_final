"""
Three test families for 0/1 Knapsack experiments.

Family 1 — Uncorrelated Random:
  Weights and values drawn independently. No structural relationship.
  This is the "easy" baseline — greedy often does well here.

Family 2 — Correlated:
  Values positively correlated with weights (v_i ≈ w_i + noise).
  Ratio-based greedy is weakened because heavy items also tend to be valuable.

Family 3 — Adversarial:
  Hand-crafted instances designed to maximize greedy's suboptimality.
  Pattern: one "bait" item with high ratio + a group of items that together
  fill the knapsack better but have lower individual ratios.
"""

import numpy as np


def generate_uncorrelated(n, W, rng):
    """Family 1: weights and values independent uniform."""
    weights = rng.randint(1, W // 2 + 1, size=n).tolist()
    values = rng.randint(1, W // 2 + 1, size=n).tolist()
    return weights, values


def generate_correlated(n, W, rng):
    """Family 2: values correlated with weights."""
    weights = rng.randint(1, W // 2 + 1, size=n)
    noise = rng.randint(-max(W // 10, 1), max(W // 10, 1) + 1, size=n)
    values = np.clip(weights + noise, 1, None).tolist()
    weights = weights.tolist()
    return weights, values


def generate_adversarial(n, W, rng):
    """Family 3: adversarial instances that punish greedy.

    Strategy: create a "jackpot + bait" structure that forces greedy to
    miss the globally optimal combination.

    - One JACKPOT item: uses 60-80% of capacity, moderate ratio (~2.0-2.5),
      but very high absolute value. Greedy won't reach it first.
    - Several BAIT items: small, high ratio (3.0-4.0). Greedy grabs these
      first. Their total weight just barely exceeds remaining capacity after
      jackpot, so jackpot + all bait won't fit — greedy must choose.
    - FILLER items: low ratio, small weight. Greedy fills remaining space
      with these after bait, getting mediocre total value.

    Greedy path: all bait + fillers (jackpot blocked by insufficient remaining capacity)
    DP optimal: jackpot + subset of bait items that fit in remaining capacity

    This reliably produces 10-25% gaps, mirroring the hand-designed 6-item example.
    """
    # Jackpot item: large, moderate ratio
    jackpot_frac = 0.6 + rng.random() * 0.2  # 60-80% of W
    jackpot_w = max(2, int(W * jackpot_frac))
    jackpot_v = max(1, int(jackpot_w * (2.0 + rng.random() * 0.5)))

    # Bait items: small, high ratio — greedy takes these first
    # Total bait weight slightly exceeds W - jackpot_w so greedy can't fit jackpot after bait
    n_bait = max(2, n // 3)
    bait_budget = (W - jackpot_w) + rng.randint(2, max(3, W // 10))
    bait_w_each = max(1, bait_budget // n_bait)

    bait_weights = []
    bait_values = []
    for _ in range(n_bait):
        w = max(1, bait_w_each + rng.randint(-1, 2))
        v = max(1, int(w * (3.0 + rng.random() * 1.0)))
        bait_weights.append(w)
        bait_values.append(v)

    # Filler items: low ratio, small
    n_filler = n - 1 - n_bait
    filler_weights = []
    filler_values = []
    for _ in range(n_filler):
        w = max(1, rng.randint(1, max(2, W // 10)))
        v = max(1, int(w * (1.0 + rng.random() * 0.5)))
        filler_weights.append(w)
        filler_values.append(v)

    weights = [jackpot_w] + bait_weights + filler_weights
    values = [jackpot_v] + bait_values + filler_values

    return weights, values


def generate_family(family_name, n, W, num_instances, seed=42):
    """Generate a batch of instances for a given test family.

    Args:
        family_name: "uncorrelated", "correlated", or "adversarial"
        n: items per instance
        W: capacity
        num_instances: how many instances
        seed: random seed

    Returns:
        list of dicts with keys: weights, values, capacity, n, family
    """
    rng = np.random.RandomState(seed)
    generators = {
        "uncorrelated": generate_uncorrelated,
        "correlated": generate_correlated,
        "adversarial": generate_adversarial,
    }
    gen = generators[family_name]

    instances = []
    for _ in range(num_instances):
        weights, values = gen(n, W, rng)
        instances.append({
            "weights": weights,
            "values": values,
            "capacity": W,
            "n": n,
            "family": family_name,
        })
    return instances
