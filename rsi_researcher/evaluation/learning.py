"""Behavioral learning diagnostics from controlled truth; not a free-text memory judge."""

import argparse
from itertools import combinations
import json
from pathlib import Path

from rsi_researcher.runtime.client import atomic_json
from rsi_researcher.evaluation.metrics import evaluate


def audit_learning(world, events, pair):
    public, private = world["public"], world["private"]
    if len(pair) != 2:
        raise ValueError("both evaluation directions are required")
    # Reuse ranking/truth validation and keep malformed model output in the denominator.
    validity = [{row["query_id"]: row["valid"] for row in evaluate(
        public["queries"], ranking, private["utilities"], private["popular"])["queries"]} for ranking in pair]
    seen = {user: set() for user in public["initial"]["users"]}
    for event in events:
        seen[event["user"]].add(event["positive"])
    rows = []
    for query in public["queries"]:
        user, qid, panel = query["user"], query["query_id"], query["candidates"]
        truth = private["utilities"][user]
        comparisons = [(a, b) if truth[a] > truth[b] else (b, a)
                       for a, b in combinations(panel, 2) if truth[a] != truth[b]]
        observed = [(a, b) for a, b in comparisons if a in seen[user] and b in seen[user]]
        correct, observed_correct = 0, 0
        for direction, rankings in enumerate(pair):
            if not validity[direction][qid]:
                continue
            positions = {item: rank for rank, item in enumerate(rankings[qid])}
            correct += sum(positions[a] < positions[b] for a, b in comparisons)
            observed_correct += sum(positions[a] < positions[b] for a, b in observed)
        rows.append({"query_id": qid, "observed_items_in_panel": len(seen[user].intersection(panel)),
                     "non_tied_pairs": len(comparisons), "observed_non_tied_pairs": len(observed),
                     "pair_accuracy": correct / (2 * len(comparisons)) if comparisons else None,
                     "observed_pair_accuracy": observed_correct / (2 * len(observed)) if observed else None,
                     "observed_pair_fraction": len(observed) / len(comparisons) if comparisons else None})
    def mean_available(key):
        values = [row[key] for row in rows if row[key] is not None]
        return sum(values) / len(values) if values else None
    return {"query_count": len(rows), "queries_with_observed_comparison": sum(row["observed_non_tied_pairs"] > 0 for row in rows),
            "mean_pair_accuracy": mean_available("pair_accuracy"),
            "mean_observed_pair_accuracy": mean_available("observed_pair_accuracy"),
            "mean_observed_pair_fraction": mean_available("observed_pair_fraction"),
            "queries": rows, "memory_coverage_guard": None,
            "limitation": "Behavioral diagnostic only. Consumption of two items does not reveal their relative preference. "
                          "No claim of free-text correctness or calibrated effective-memory coverage."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="controlled-world workflow run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.run), Path(args.output)
    if output.exists():
        raise ValueError("output already exists")
    world = json.loads((source / "world.json").read_text(encoding="utf-8"))
    report = {"purpose": "offline development diagnostics; zero model calls", "arms": {}}
    for arm in ("B0", "candidate"):
        saved = json.loads((source / arm / "rankings.json").read_text(encoding="utf-8"))
        report["arms"][arm] = audit_learning(world, world["public"]["events"]["high"], [saved["forward"], saved["reverse"]])
    atomic_json(output, report)
