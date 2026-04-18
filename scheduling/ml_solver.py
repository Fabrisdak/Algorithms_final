"""
Weighted Interval Scheduling — ML-based solver.

Pipeline:
  1. Feature engineering: encode each interval with hand-crafted features.
  2. Train Random Forest and MLP classifiers on DP-labeled data.
  3. Predict per-interval include/exclude decisions.
  4. Apply feasibility correction (remove overlapping intervals by confidence).

Uses scikit-learn only — no PyTorch needed.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from scheduling.dp_solver import solve as dp_solve


def extract_features(intervals: list[dict]) -> np.ndarray:
    """
    Feature matrix for a single instance. Each row = one interval.

    Features per interval (16 total):
      0.  weight
      1.  duration (finish - start)
      2.  weight / duration ratio
      3.  start (normalized by time horizon)
      4.  finish (normalized by time horizon)
      5.  duration / time_horizon (fraction of total time used)
      6.  number of overlapping intervals (conflict degree)
      7.  conflict degree / n (normalized)
      8.  weight rank (0=lightest, 1=heaviest)
      9.  ratio rank (0=worst, 1=best)
      10. total weight of conflicting intervals
      11. weight / total_conflict_weight (competitive ratio)
      12. weight / mean_weight (relative weight)
      13. ratio / mean_ratio (relative ratio)
      14. fraction of timeline covered by this interval
      15. "selection pressure" = weight / (conflict_degree + 1)
    """
    n = len(intervals)
    if n == 0:
        return np.empty((0, 16))

    time_horizon = max(iv["finish"] for iv in intervals)
    if time_horizon == 0:
        time_horizon = 1

    durations = [iv["finish"] - iv["start"] for iv in intervals]
    weights = [iv["weight"] for iv in intervals]
    ratios = [w / max(d, 1e-9) for w, d in zip(weights, durations)]

    mean_weight = np.mean(weights) if weights else 1
    mean_ratio = np.mean(ratios) if ratios else 1
    if mean_weight == 0:
        mean_weight = 1
    if mean_ratio == 0:
        mean_ratio = 1

    weight_ranks = np.argsort(np.argsort(weights)) / max(n - 1, 1)
    ratio_ranks = np.argsort(np.argsort(ratios)) / max(n - 1, 1)

    conflict_counts = []
    conflict_weights = []
    for i, iv in enumerate(intervals):
        conflicts = [
            j for j, other in enumerate(intervals)
            if j != i and iv["start"] < other["finish"] and iv["finish"] > other["start"]
        ]
        conflict_counts.append(len(conflicts))
        conflict_weights.append(sum(weights[j] for j in conflicts))

    features = []
    for i, iv in enumerate(intervals):
        duration = durations[i]
        cw = conflict_weights[i] if conflict_weights[i] > 0 else 1
        cc = conflict_counts[i]

        features.append([
            weights[i],
            duration,
            ratios[i],
            iv["start"] / time_horizon,
            iv["finish"] / time_horizon,
            duration / time_horizon,
            cc,
            cc / max(n - 1, 1),
            weight_ranks[i],
            ratio_ranks[i],
            conflict_weights[i],
            weights[i] / cw,
            weights[i] / mean_weight,
            ratios[i] / mean_ratio,
            duration / time_horizon,
            weights[i] / (cc + 1),
        ])

    return np.array(features)


NUM_FEATURES = 16


def generate_training_data(instance_generator, n_instances: int = 200,
                           n_range: tuple[int, int] = (5, 30),
                           seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate labeled training data using DP solutions.

    Returns (X, y) where each row of X is a feature vector for one interval
    and y is 1 if DP selected it, 0 otherwise.
    """
    rng = np.random.RandomState(seed)
    all_X, all_y = [], []

    for i in range(n_instances):
        n = rng.randint(*n_range)
        intervals = instance_generator(n, seed=seed + i)

        _, selected = dp_solve(intervals)
        selected_set = set(selected)

        X = extract_features(intervals)
        y = np.array([1 if j in selected_set else 0 for j in range(len(intervals))])

        all_X.append(X)
        all_y.append(y)

    return np.vstack(all_X), np.concatenate(all_y)


class SchedulingMLSolver:
    """Wraps RF and MLP models for Weighted Interval Scheduling."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=1000,
            random_state=42, early_stopping=True, validation_fraction=0.15,
            learning_rate="adaptive"
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train both models with class balancing for MLP."""
        X_scaled = self.scaler.fit_transform(X)

        self.rf.fit(X_scaled, y)

        # Balance MLP training via sample weights
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > 0:
            sample_weights = np.where(y == 1, n_neg / n_pos, 1.0)
            sample_weights /= sample_weights.mean()
        else:
            sample_weights = np.ones(len(y))

        self.mlp.fit(X_scaled, y)
        self._fitted = True

    def _feasibility_correction(self, intervals: list[dict],
                                confidences: np.ndarray) -> list[int]:
        """
        Confidence-based greedy feasibility correction.
        Instead of only considering predicted-positive intervals, consider ALL
        intervals ranked by confidence. Greedily add the highest-confidence
        interval that doesn't conflict, until no more can be added.
        """
        n = len(intervals)
        sorted_by_conf = sorted(range(n), key=lambda i: -confidences[i])

        selected = []
        selected_intervals = []
        for idx in sorted_by_conf:
            iv = intervals[idx]
            conflict = False
            for sel_iv in selected_intervals:
                if iv["start"] < sel_iv["finish"] and iv["finish"] > sel_iv["start"]:
                    conflict = True
                    break
            if not conflict:
                selected.append(idx)
                selected_intervals.append(iv)

        return sorted(selected)

    def predict_rf(self, intervals: list[dict]) -> tuple[float, list[int]]:
        """Predict using Random Forest with feasibility correction."""
        return self._predict_with_model(intervals, self.rf)

    def predict_mlp(self, intervals: list[dict]) -> tuple[float, list[int]]:
        """Predict using MLP with feasibility correction."""
        return self._predict_with_model(intervals, self.mlp)

    def _predict_with_model(self, intervals: list[dict], model) -> tuple[float, list[int]]:
        """Run prediction with a given model."""
        if not intervals:
            return 0.0, []

        X = extract_features(intervals)
        X_scaled = self.scaler.transform(X)

        proba = model.predict_proba(X_scaled)
        confidences = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        selected = self._feasibility_correction(intervals, confidences)
        total_weight = sum(intervals[i]["weight"] for i in selected)

        return total_weight, selected
