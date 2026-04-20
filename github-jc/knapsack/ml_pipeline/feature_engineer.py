"""
Per-item feature extraction for ML-based knapsack solving.

Each item is represented by a feature vector used to predict whether
the item should be included in the optimal solution (binary classification).

Feature list:
  1. value           — item value (raw)
  2. weight          — item weight (raw)
  3. vw_ratio        — value / weight
  4. weight_frac     — weight / capacity (how much of knapsack this item uses)
  5. value_frac      — value / max_value (relative value)
  6. ratio_rank      — rank by v/w ratio (0 = best ratio, normalized to [0,1])
  7. weight_rank     — rank by weight (0 = lightest, normalized to [0,1])
  8. value_rank      — rank by value (0 = most valuable, normalized to [0,1])
"""

import numpy as np


FEATURE_NAMES = [
    "value", "weight", "vw_ratio", "weight_frac",
    "value_frac", "ratio_rank", "weight_rank", "value_rank",
]


def extract_features(instance):
    """Extract per-item features from a knapsack instance.

    Args:
        instance: dict with keys weights, values, capacity, n

    Returns:
        np.ndarray of shape (n, num_features)
    """
    weights = np.array(instance["weights"], dtype=float)
    values = np.array(instance["values"], dtype=float)
    capacity = instance["capacity"]
    n = instance["n"]

    vw_ratio = values / np.maximum(weights, 1e-9)
    weight_frac = weights / capacity
    max_val = values.max() if values.max() > 0 else 1.0
    value_frac = values / max_val

    # Ranks normalized to [0, 1]: 0 = best
    ratio_rank = np.argsort(np.argsort(-vw_ratio)).astype(float) / max(n - 1, 1)
    weight_rank = np.argsort(np.argsort(weights)).astype(float) / max(n - 1, 1)
    value_rank = np.argsort(np.argsort(-values)).astype(float) / max(n - 1, 1)

    features = np.column_stack([
        values, weights, vw_ratio, weight_frac,
        value_frac, ratio_rank, weight_rank, value_rank,
    ])

    return features
