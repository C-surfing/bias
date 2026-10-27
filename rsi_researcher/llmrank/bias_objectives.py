"""Registered, label-blind scientific decision contracts for LLMRank debias.

Jiang et al., Beyond Utility (WWW 2025), Appendix A.1:
PopREO (five popularity strata) and Candidate Position Bias (paired
target-first vs random presentations). ARP/APLT are exposure proxies,
NOT interchangeable substitutes for either primary bias outcome.
"""
from __future__ import annotations

CONTRACT = "llmrank_target_bias_objectives_v1"
OBJECTIVES = {
    "popularity": {
        "primary_metric": "PopREO@5",
        "min_group_positives": 5,
        "min_absolute_reduction": 0.02,
        "proxy_metrics": ("ARP@5", "APLT@5"),
        "scope": "observed-positive opportunity parity; not causal popularity identification",
    },
    "candidate_position": {
        "primary_metric": "CandidatePositionBias_NDCG@5",
        "min_paired_users": 10,
        "min_absolute_reduction": 0.02,
        "proxy_metrics": ("CandidatePositionBias_HR@5",),
        "scope": "target-first versus randomized candidate presentation, not logged click exposure",
    },
}


def evaluate_target(baseline, candidate, bias_target):
    """Return a primary-target verdict without looking at unrelated proxies."""
    if bias_target not in OBJECTIVES:
        raise ValueError(f"unsupported LLMRank bias target: {bias_target}")
    obj = OBJECTIVES[bias_target]
    key = obj["primary_metric"]
    before, after = baseline.get(key), candidate.get(key)
    support = {}
    if bias_target == "popularity":
        left = baseline.get("popreo_group_positive_counts")
        right = candidate.get("popreo_group_positive_counts")
        floor = obj["min_group_positives"]
        support = {"baseline_group_positives": left,
                   "candidate_group_positives": right,
                   "minimum_positives_per_group": floor}
        supported = (
            isinstance(left, list) and isinstance(right, list)
            and len(left) == len(right) == 5 and left == right
            and all(type(n) is int and n >= floor for n in left)
        )
    else:
        floor = obj["min_paired_users"]
        support = {"baseline_users": baseline.get("users"),
                   "candidate_users": candidate.get("users"),
                   "minimum_paired_users": floor}
        supported = (
            isinstance(baseline.get("users"), int)
            and baseline["users"] >= floor
            and baseline["users"] == candidate.get("users")
        )
    import math
    defined = (before is not None and after is not None
               and isinstance(before, (int, float))
               and isinstance(after, (int, float))
               and math.isfinite(before) and math.isfinite(after))
    if defined:
        before = float(before)
        after = float(after)
    gain = ((before - after) if bias_target == "popularity"
            else (abs(before) - abs(after))) if defined else None
    ready = supported and defined
    return {
        "target": bias_target, "metric": key,
        "baseline": before, "candidate": after,
        "reduction": gain, "threshold": obj["min_absolute_reduction"],
        "supported": bool(ready),
        "improved": bool(ready and gain > obj["min_absolute_reduction"]),
        "support": support, "scope": obj["scope"],
        "status": ("insufficient_target_bias_support" if not ready
                   else "target_bias_improved" if gain > obj["min_absolute_reduction"]
                   else "target_bias_not_improved"),
        "causal_confirmation": False,
    }



def bootstrap_target_bias(baseline, candidate, popularity, catalog, bias_target,
                          *, seed=20261021, draws=1000):
    """Paired user bootstrap of target bias reduction (positive is better).

    Resampled panels retain the same user/candidate and first/random arms.
    Undefined or unsupported resamples are NOT silently replaced by zero.
    """
    import random
    from .dataset import summarize, cpb
    rows_a,rows_b=baseline["random"],candidate["random"]
    first_a,first_b=baseline["first"],candidate["first"]
    n=len(rows_a)
    if (not n or any(len(rows)!=n for rows in (rows_b,first_a,first_b))
        or any(len(set(r["user"] for r in rows))!=n
               for rows in (rows_a,rows_b,first_a,first_b))
        or any(tuple(r["user"] for r in rows)!=tuple(r["user"] for r in rows_a)
               for rows in (rows_b,first_a,first_b))):
        raise ValueError("target bootstrap requires aligned unique users")
    rng=random.Random(seed)
    gains=[]
    for _ in range(draws):
        ix=[rng.randrange(n) for _ in range(n)]
        def sample(rows,first):
            m=summarize([rows[i] for i in ix],popularity,catalog,5)
            if bias_target=="candidate_position":
                other=summarize([first[i] for i in ix],popularity,catalog,5)
                m["CandidatePositionBias_NDCG@5"]=cpb(other["NDCG@5"],
                                                         m["NDCG@5"])
            return m
        measured=evaluate_target(sample(rows_a,first_a),sample(rows_b,first_b),
                                 bias_target)
        if measured["supported"]:
            gains.append(measured["reduction"])
    gains.sort()
    supported=len(gains)>=int(0.8*draws)
    return {"bias_target":bias_target,"direction":"positive_reduction_is_better",
            "paired_users":n,"valid_resamples":len(gains),
            "requested_resamples":draws,
            "ci95":([gains[int(len(gains)*.025)],
                     gains[min(len(gains)-1,int(len(gains)*.975))]]
                    if supported and gains else None),
            "inference":"descriptive paired bootstrap; no sequential error control"}
