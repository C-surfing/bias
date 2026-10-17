"""Metrics for the author's sampled candidates with one known target."""
import math

from rsi_researcher.datasets.author import popularity_group


def ndcg(ranking, target, k=10):
    return 1.0 / math.log2(ranking.index(target) + 2) if target in ranking[:k] else 0.0


def score_author_ranking(ranking, query, profile):
    target = query["target"]
    group = lambda item: popularity_group(item, profile["training_item_counts"],
                                         profile["head_min_count"], profile["tail_max_count"])
    top = ranking[:5]
    candidates = query["candidates"]
    top5_head = sum(group(i) == "head" for i in top) / 5
    candidate_head = sum(group(i) == "head" for i in candidates) / len(candidates)
    return {"ndcg_at_5": ndcg(ranking, target, 5), "ndcg_at_10": ndcg(ranking, target, 10),
            "hit_at_1": int(target in ranking[:1]), "hit_at_5": int(target in top),
            "target_group": group(target),
            "top5_head_share": top5_head,
            "candidate_head_share": candidate_head,
            "excess_popularity_exposure": top5_head - candidate_head}


def summarize_author_rankings(rows):
    metrics = ("ndcg_at_5", "ndcg_at_10", "hit_at_1", "hit_at_5", "top5_head_share",
               "candidate_head_share", "excess_popularity_exposure")
    summary = {"ranking_count": len(rows), "invalid_rankings": sum(not row["valid"] for row in rows),
               **{"mean_" + key: sum(row[key] for row in rows) / len(rows) for key in metrics}}
    summary["target_groups"] = {}
    for group in ("head", "mid", "tail", "unseen"):
        selected = [row for row in rows if row["target_group"] == group]
        summary["target_groups"][group] = {"count": len(selected),
            "hit_at_5": sum(row["hit_at_5"] for row in selected) / len(selected) if selected else None}
    summary["popularity_interpretation"] = "Exposure descriptions only; candidate mix and missing relevance prevent a causal bias conclusion."
    return summary
