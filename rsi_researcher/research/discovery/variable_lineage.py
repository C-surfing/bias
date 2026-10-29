"""Pre-outcome construct lineage checks for discovery experiment proposals.

This is a deliberately narrow dependency audit, not a causal graph or an
automatic claim of statistical independence. Known feature windows come from
dataset preparation; unregistered expressions are *unknown*, not assumed safe.
"""
from __future__ import annotations

import re


def _lineage(tools, column: str, seen: frozenset[str] = frozenset()) -> list[tuple[str, int, int]]:
    """(source outcome, first lag, last lag), with lag=0 at event time."""
    if column in seen:
        return []
    catalog = getattr(tools.adapter, "feature_lineage", {})
    if column in catalog:
        source, start, end = catalog[column]
        return [(source, int(start), int(end))]
    derived = getattr(tools, "feature_records", {}).get(column)
    if not derived:
        return []
    # SQL scalar derived groups can contain several recorded input fields.
    # Never assign temporal lineage to unknown identifiers.
    expr = derived.get("expression", "")
    tokens = set(re.findall(r"\b[a-z][a-z0-9_]*\b", expr.lower()))
    result = []
    for field in sorted(tokens):
        if field == column:
            continue
        result.extend(_lineage(tools, field, seen | {column}))
    return result


def audit_construct_overlap(tools, predictor: str, outcome: str) -> dict:
    left = _lineage(tools, predictor)
    right = _lineage(tools, outcome)
    shared = []
    if predictor == outcome:
        shared.append({"source": predictor, "lags": "identical columns"})
    for source1, lo1, hi1 in left:
        for source2, lo2, hi2 in right:
            if source1 == source2 and max(lo1, lo2) <= min(hi1, hi2):
                shared.append({
                    "source": source1,
                    "overlap_lags": [max(lo1, lo2), min(hi1, hi2)],
                })
    return {
        "predictor": predictor, "outcome": outcome,
        "predictor_lineage": left, "outcome_lineage": right,
        "status": "overlapping_constructs" if shared else (
            "lineage_known_no_overlap" if left and right else
            "lineage_unverified"),
        "shared": shared,
    }


def audit_observation_sql(tools, sql: str) -> dict | None:
    """Detect a known aggregate/history tautology in an exploratory query.

    A full SQL parser is not attempted: only recognized aggregate targets and
    referenced registered source fields can trigger a *blocking* finding.
    """
    aggregates = set(re.findall(
        r"\b(?:avg|mean|sum|min|max)\s*\(\s*(?:cast\s*\(\s*)?"
        r"([a-z][a-z0-9_]*)\b", sql, flags=re.I))
    columns = set(re.findall(r"\b[a-z][a-z0-9_]*\b", sql.lower()))
    catalog = set(getattr(tools.adapter, "feature_lineage", {}))
    catalog.update(getattr(tools, "feature_records", {}))
    for outcome in sorted(aggregates):
        for predictor in sorted(columns & catalog - {outcome}):
            result = audit_construct_overlap(tools, predictor, outcome)
            if result["status"] == "overlapping_constructs":
                return result
    return None
