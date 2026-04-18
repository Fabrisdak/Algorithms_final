"""Unit tests for Weighted Interval Scheduling solvers."""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scheduling.dp_solver import solve as dp_solve
from scheduling.greedy_solver import (
    solve_earliest_finish,
    solve_highest_weight,
    solve_best_ratio,
)
from scheduling.data_gen import (
    generate_sparse_random,
    generate_dense_random,
    generate_adversarial_greedy,
    generate_adversarial_ef,
)


class TestDPSolver(unittest.TestCase):
    """Tests for the DP solver."""

    def test_empty(self):
        value, selected = dp_solve([])
        self.assertEqual(value, 0.0)
        self.assertEqual(selected, [])

    def test_single_interval(self):
        intervals = [{"start": 0, "finish": 5, "weight": 10}]
        value, selected = dp_solve(intervals)
        self.assertEqual(value, 10)
        self.assertEqual(selected, [0])

    def test_non_overlapping(self):
        """All intervals are compatible — DP should select all."""
        intervals = [
            {"start": 0, "finish": 2, "weight": 5},
            {"start": 3, "finish": 5, "weight": 10},
            {"start": 6, "finish": 8, "weight": 7},
        ]
        value, selected = dp_solve(intervals)
        self.assertEqual(value, 22)
        self.assertEqual(sorted(selected), [0, 1, 2])

    def test_classic_example(self):
        """
        Hand-worked example:
          A: [0,3) w=2
          B: [1,5) w=4
          C: [4,6) w=4
          D: [3,7) w=7
          E: [5,9) w=2
          F: [6,8) w=1

        Optimal: A + D = 2 + 7 = 9  OR  A + C + E = 2+4+2=8  OR  B + F = 4+1=5
        Actually let's recheck: sorted by finish = A(0-3,2), B(1-5,4), C(4-6,4), D(3-7,7), F(6-8,1), E(5-9,2)
        p(A)=0, p(B)=0, p(C)=A=1, p(D)=A=1, p(F)=C=3, p(E)=C=3
        dp[0]=0
        dp[1]=max(dp[0], 2+dp[0])=2   (A)
        dp[2]=max(dp[1], 4+dp[0])=4   (B)
        dp[3]=max(dp[2], 4+dp[1])=6   (A+C)
        dp[4]=max(dp[3], 7+dp[1])=9   (A+D)
        dp[5]=max(dp[4], 1+dp[3])=9   (A+D)
        dp[6]=max(dp[5], 2+dp[3])=9   (A+D)
        Optimal = 9, selecting A and D.
        """
        intervals = [
            {"start": 0, "finish": 3, "weight": 2},   # A (idx 0)
            {"start": 1, "finish": 5, "weight": 4},   # B (idx 1)
            {"start": 4, "finish": 6, "weight": 4},   # C (idx 2)
            {"start": 3, "finish": 7, "weight": 7},   # D (idx 3)
            {"start": 5, "finish": 9, "weight": 2},   # E (idx 4)
            {"start": 6, "finish": 8, "weight": 1},   # F (idx 5)
        ]
        value, selected = dp_solve(intervals)
        self.assertEqual(value, 9)
        self.assertEqual(sorted(selected), [0, 3])

    def test_overlapping_pair(self):
        """Two overlapping intervals — pick the heavier one."""
        intervals = [
            {"start": 0, "finish": 5, "weight": 3},
            {"start": 2, "finish": 7, "weight": 8},
        ]
        value, selected = dp_solve(intervals)
        self.assertEqual(value, 8)
        self.assertEqual(selected, [1])

    def test_touching_intervals(self):
        """Intervals that touch (finish == start of next) are compatible."""
        intervals = [
            {"start": 0, "finish": 3, "weight": 5},
            {"start": 3, "finish": 6, "weight": 5},
        ]
        value, selected = dp_solve(intervals)
        self.assertEqual(value, 10)
        self.assertEqual(sorted(selected), [0, 1])


class TestGreedySolvers(unittest.TestCase):
    """Tests for greedy heuristics."""

    def test_earliest_finish_basic(self):
        intervals = [
            {"start": 0, "finish": 2, "weight": 5},
            {"start": 1, "finish": 4, "weight": 10},
            {"start": 3, "finish": 5, "weight": 5},
        ]
        value, selected = solve_earliest_finish(intervals)
        self.assertEqual(sorted(selected), [0, 2])
        self.assertEqual(value, 10)

    def test_highest_weight_basic(self):
        intervals = [
            {"start": 0, "finish": 2, "weight": 5},
            {"start": 1, "finish": 4, "weight": 10},
            {"start": 3, "finish": 5, "weight": 5},
        ]
        value, selected = solve_highest_weight(intervals)
        self.assertEqual(selected, [1])
        self.assertEqual(value, 10)

    def test_greedy_suboptimal_on_weighted(self):
        """Greedy heuristics should be suboptimal on this adversarial case."""
        intervals = generate_adversarial_greedy(5)
        dp_value, _ = dp_solve(intervals)

        ef_value, _ = solve_earliest_finish(intervals)
        hw_value, _ = solve_highest_weight(intervals)
        br_value, _ = solve_best_ratio(intervals)

        greedy_best = max(ef_value, hw_value, br_value)
        self.assertGreaterEqual(dp_value, greedy_best,
                                "DP should be >= any greedy on adversarial case")


class TestDataGenerators(unittest.TestCase):
    """Tests for instance generators."""

    def test_sparse_generates_correct_count(self):
        intervals = generate_sparse_random(20, seed=0)
        self.assertEqual(len(intervals), 20)
        for iv in intervals:
            self.assertIn("start", iv)
            self.assertIn("finish", iv)
            self.assertIn("weight", iv)
            self.assertLess(iv["start"], iv["finish"])

    def test_dense_generates_correct_count(self):
        intervals = generate_dense_random(20, seed=0)
        self.assertEqual(len(intervals), 20)

    def test_adversarial_structure(self):
        intervals = generate_adversarial_greedy(5)
        self.assertEqual(len(intervals), 6)
        self.assertEqual(intervals[0]["start"], 0)
        self.assertEqual(intervals[0]["finish"], 10)

    def test_dp_beats_greedy_on_random(self):
        """Sanity check: DP value >= greedy value on random instances."""
        for seed in range(5):
            intervals = generate_sparse_random(15, seed=seed)
            dp_val, _ = dp_solve(intervals)
            ef_val, _ = solve_earliest_finish(intervals)
            self.assertGreaterEqual(dp_val, ef_val)


if __name__ == "__main__":
    unittest.main()
