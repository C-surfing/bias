"""External controlled-world evaluation; never include hidden utility in model prompts."""

import math


def evaluate(queries, rankings, utilities, popular):
    """Each query has a fixed >=10-item panel; utility gains are linear, nonnegative."""
    if not queries:
        raise ValueError("evaluation needs queries")
    ids = [q["query_id"] for q in queries]
    if len(set(ids)) != len(ids) or set(rankings) - set(ids):
        raise ValueError("duplicate query or unexpected ranking")
    discounts = [1 / math.log2(j + 2) for j in range(10)]
    total_discount = sum(discounts)
    results = []
    for query in queries:
        qid, user, panel = query["query_id"], query["user"], query["candidates"]
        if len(panel) < 10 or len(set(panel)) != len(panel):
            raise ValueError("panel needs at least ten distinct items")
        truth = utilities[user]
        if any(type(truth[i]) not in (int, float) or not math.isfinite(truth[i]) or truth[i] < 0 for i in panel):
            raise ValueError("utility must be finite and nonnegative")
        ideal = sorted(panel, key=lambda i: (-truth[i], i))
        ideal_dcg = sum(d * truth[i] for d, i in zip(discounts, ideal))
        if ideal_dcg <= 0:
            raise ValueError("zero ideal utility: relative quality is undefined")
        ideal_head = sum(d for d, i in zip(discounts, ideal) if i in popular) / total_discount
        ranking = rankings.get(qid)
        valid = (isinstance(ranking, list) and all(isinstance(i, str) for i in ranking)
                 and len(ranking) == len(panel) and set(ranking) == set(panel))
        if valid:
            quality = sum(d * truth[i] for d, i in zip(discounts, ranking)) / ideal_dcg
            head = sum(d for d, i in zip(discounts, ranking) if i in popular) / total_discount
        else:
            # Invalid output must not earn a debiasing reward by returning no popular items.
            quality, head = 0.0, 1.0
        results.append({"query_id": qid, "valid": valid, "ndcg_at_10": quality,
                        "head_share": head, "ideal_head_share": ideal_head,
                        "signed_excess": head - ideal_head,
                        "bias": max(0.0, head - ideal_head)})
    n = len(results)
    return {"query_count": n, "invalid_rankings": sum(not r["valid"] for r in results),
            "utility": sum(r["ndcg_at_10"] for r in results) / n,
            "bias": sum(r["bias"] for r in results) / n,
            "signed_excess": sum(r["signed_excess"] for r in results) / n,
            "queries": results}


def point_decision(baseline, candidate, guards, minimum_bias):
    """Exploratory gate only. Always pass initial B0, never the latest parent."""
    required = {"position", "memory_coverage", "learning", "budget"}
    if set(guards) != required or any(v is not None and type(v) is not bool for v in guards.values()):
        raise ValueError("all fixed guards must be supplied as true, false, or unknown")
    if not math.isfinite(minimum_bias) or minimum_bias <= 0:
        raise ValueError("freeze a positive measurable-bias floor before comparing")
    for result in (baseline, candidate):
        if any(type(result[k]) not in (int, float) or not math.isfinite(result[k]) or not 0 <= result[k] <= 1
               for k in ("utility", "bias")):
            raise ValueError("invalid evaluation metric")
        if type(result["query_count"]) is not int or result["query_count"] <= 0:
            raise ValueError("invalid query count")
        if type(result["invalid_rankings"]) is not int or not 0 <= result["invalid_rankings"] <= result["query_count"]:
            raise ValueError("invalid failure count")
    if baseline["query_count"] != candidate["query_count"]:
        raise ValueError("evaluation denominator changed")
    if candidate["invalid_rankings"] or False in guards.values():
        return {"status": "rejected", "reason": "candidate output or independent guard failed",
                "guards": dict(guards), "statistical_support": "not_tested"}
    if baseline["utility"] == 0 or baseline["bias"] < minimum_bias or baseline["invalid_rankings"]:
        return {"status": "inconclusive", "reason": "baseline cannot support relative comparison",
                "guards": dict(guards), "statistical_support": "not_tested"}
    quality_margin = candidate["utility"] - 0.99 * baseline["utility"]
    bias_margin = 0.90 * baseline["bias"] - candidate["bias"]
    result = {"quality_margin": quality_margin, "bias_margin": bias_margin,
              "guards": dict(guards), "statistical_support": "not_tested"}
    if quality_margin < 0 or bias_margin < 0 or candidate["invalid_rankings"] or False in guards.values():
        return result | {"status": "rejected", "reason": "at least one independent gate failed"}
    if None in guards.values():
        return result | {"status": "inconclusive", "reason": "guard evidence is incomplete"}
    return result | {"status": "eligible", "reason": "exploratory point gates only; no statistical promotion"}
