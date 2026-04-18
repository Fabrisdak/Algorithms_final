"""
Weighted Interval Scheduling — Greedy heuristics.

Three heuristics, none of which are optimal for the weighted case:
  1. Earliest Finish (EF) — classic greedy for the unweighted variant.
  2. Highest Weight (HW) — greedily pick the heaviest remaining compatible interval.
  3. Best Ratio (BR) — pick by weight / duration ratio, breaking ties by earliest finish.

Complexity: O(n log n) for the sort + O(n) scan = O(n log n) each.
"""


def _is_compatible(selected: list[dict], candidate: dict) -> bool:
    """Check if candidate doesn't overlap with any selected interval."""
    for s in selected:
        if candidate["start"] < s["finish"] and candidate["finish"] > s["start"]:
            return False
    return True


def _greedy_template(intervals: list[dict], key_fn, reverse: bool = False) -> tuple[float, list[int]]:
    """
    Generic greedy: sort by key_fn, iterate, pick if compatible.
    Returns (total_weight, list_of_original_indices).
    """
    if not intervals:
        return 0.0, []

    indexed = [(iv, i) for i, iv in enumerate(intervals)]
    indexed.sort(key=lambda x: key_fn(x[0]), reverse=reverse)

    selected = []
    selected_indices = []
    total_weight = 0.0

    for iv, orig_idx in indexed:
        if _is_compatible(selected, iv):
            selected.append(iv)
            selected_indices.append(orig_idx)
            total_weight += iv["weight"]

    return total_weight, sorted(selected_indices)


def solve_earliest_finish(intervals: list[dict]) -> tuple[float, list[int]]:
    """Greedy by earliest finish time (optimal for unweighted, not for weighted)."""
    return _greedy_template(intervals, key_fn=lambda iv: iv["finish"])


def solve_highest_weight(intervals: list[dict]) -> tuple[float, list[int]]:
    """Greedy by highest weight first."""
    return _greedy_template(intervals, key_fn=lambda iv: iv["weight"], reverse=True)


def solve_best_ratio(intervals: list[dict]) -> tuple[float, list[int]]:
    """Greedy by weight / duration ratio (highest first), tie-break by earliest finish."""
    def ratio_key(iv):
        duration = iv["finish"] - iv["start"]
        if duration == 0:
            return float("inf")
        return iv["weight"] / duration
    return _greedy_template(
        intervals,
        key_fn=lambda iv: (-ratio_key(iv), iv["finish"]),
        reverse=False,
    )
