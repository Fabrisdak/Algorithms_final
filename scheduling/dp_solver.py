"""
Weighted Interval Scheduling — Dynamic Programming solver.

Algorithm:
  1. Sort intervals by finish time.
  2. For each interval i, compute p(i) = index of the latest interval
     that finishes before interval i starts (binary search).
  3. Build table: dp[i] = max(dp[i-1], weight_i + dp[p(i)])
  4. Traceback to recover the selected intervals.

Complexity: O(n log n)
"""

from bisect import bisect_right


def _find_latest_compatible(intervals: list[dict], index: int) -> int:
    """
    Binary search for the latest interval that finishes <= intervals[index]['start'].
    intervals must be sorted by finish time.
    Returns the 1-based index (0 means no compatible interval).
    """
    target = intervals[index]["start"]
    finishes = [intervals[i]["finish"] for i in range(index)]
    pos = bisect_right(finishes, target)
    return pos


def solve(intervals: list[dict]) -> tuple[float, list[int]]:
    """
    Solve Weighted Interval Scheduling via DP.

    Args:
        intervals: list of dicts with keys 'start', 'finish', 'weight'.
                   Original indices are tracked automatically.

    Returns:
        (optimal_total_weight, selected_original_indices)
    """
    if not intervals:
        return 0.0, []

    n = len(intervals)
    indexed = [(iv["start"], iv["finish"], iv["weight"], i) for i, iv in enumerate(intervals)]
    indexed.sort(key=lambda x: x[1])

    sorted_intervals = [{"start": s, "finish": f, "weight": w, "orig_idx": idx}
                        for s, f, w, idx in indexed]

    p = [0] * n
    for i in range(n):
        p[i] = _find_latest_compatible(sorted_intervals, i)

    dp = [0.0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = max(dp[i - 1], sorted_intervals[i - 1]["weight"] + dp[p[i - 1]])

    selected = []
    i = n
    while i > 0:
        if sorted_intervals[i - 1]["weight"] + dp[p[i - 1]] >= dp[i - 1]:
            selected.append(sorted_intervals[i - 1]["orig_idx"])
            i = p[i - 1]
        else:
            i -= 1

    selected.reverse()
    return dp[n], selected
