"""Auxiliary popularity-reference sensitivity; never replaces frozen ID-tied scores."""

from itertools import groupby
import math

from rsi_researcher.evaluation.metrics import evaluate


def ideal_head_range(panel, utility, popular):
    ordered = sorted(panel, key=lambda i: (-utility[i], i))
    discounts = [1 / math.log2(j + 2) for j in range(10)]
    denominator = sum(discounts)
    position, low, expected, high = 0, 0.0, 0.0, 0.0
    for _, entries in groupby(ordered, key=lambda i: utility[i]):
        group = list(entries)
        slots = discounts[position:min(position + len(group), 10)]
        if not slots:
            break
        heads = sum(i in popular for i in group)
        forced = max(0, heads - (len(group) - len(slots)))
        low += sum(slots[-forced:]) if forced else 0
        high += sum(slots[:min(heads, len(slots))])
        expected += sum(slots) * heads / len(group)
        position += len(group)
    return {"minimum": low / denominator, "random_tie_expectation": expected / denominator,
            "maximum": high / denominator}


def audit_ties(world, pair):
    if len(pair) != 2:
        raise ValueError("both ranking directions are required")
    queries, private = world["public"]["queries"], world["private"]
    rows = []
    for direction, rankings in enumerate(pair):
        scored = evaluate(queries, rankings, private["utilities"], private["popular"])
        for query, result in zip(queries, scored["queries"]):
            bounds = ideal_head_range(query["candidates"], private["utilities"][query["user"]], private["popular"])
            rows.append({"query_id": query["query_id"], "direction": direction, "valid": result["valid"],
                         "primary_bias": result["bias"], "ideal_head_range": bounds,
                         "bias_lower_under_equal_utility_ties": max(0, result["head_share"] - bounds["maximum"]),
                         "bias_upper_under_equal_utility_ties": max(0, result["head_share"] - bounds["minimum"]),
                         "bias_against_random_tie_reference": max(0, result["head_share"] - bounds["random_tie_expectation"])})
    keys = ("primary_bias", "bias_lower_under_equal_utility_ties", "bias_upper_under_equal_utility_ties", "bias_against_random_tie_reference")
    return {"means": {key: sum(row[key] for row in rows) / len(rows) for key in keys}, "queries": rows,
            "interpretation": "Auxiliary reference sensitivity only; frozen primary metric and decisions are unchanged."}
