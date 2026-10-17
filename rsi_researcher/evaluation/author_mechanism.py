"""Paired development comparisons; popularity references remain observational proxies."""
import random
import re
from pathlib import Path
import json

from rsi_researcher.datasets.author import popularity_group
from rsi_researcher.evaluation.author import score_author_ranking


def measured_rows(rankings, inputs, profile):
    histories = {u: [] for u in inputs["initial"]["users"]}
    counts = profile["training_item_counts"]
    group = lambda i: popularity_group(i, counts, profile["head_min_count"], profile["tail_max_count"])
    # Replay changes learning weights, not the user's observed history.
    for event in inputs.get("source_events", inputs["events"]):
        histories[event["user"]].append(event["positive"])
    result = []
    for row in rankings:
        if not row["valid"]:
            raise ValueError("mechanism analysis requires valid rankings; fix the failed execution first")
        query = row["query"]
        values = score_author_ranking(row["ranking"], query, profile)
        user_history = histories[query["user"]]
        history_head = sum(group(i) == "head" for i in user_history) / len(user_history)
        result.append({"query_id": query["query_id"], "user": query["user"], **values,
                       "head_excess": max(0, values["top5_head_share"] - values["candidate_head_share"]),
                       "history_head_share": history_head,
                       "popularity_calibration_gap": abs(values["top5_head_share"] - history_head)})
    return result


def compare_rows(baseline, candidate, config):
    before = {r["query_id"]: r for r in baseline}
    after = {r["query_id"]: r for r in candidate}
    if len(before) != len(baseline) or before.keys() != after.keys() or len(after) != len(candidate):
        raise ValueError("paired query membership differs")
    keys = ("ndcg_at_5", "hit_at_1", "hit_at_5", "head_excess", "top5_head_share", "popularity_calibration_gap")
    mean = lambda rows, key: sum(r[key] for r in rows) / len(rows)
    values = {}
    for key in keys:
        b, c = mean(baseline, key), mean(candidate, key)
        values[key] = {"baseline": b, "candidate": c, "abs_diff": c-b, "relative_diff": (c-b)/b if b else None}
    quality, bias = values["ndcg_at_5"], values["head_excess"]
    quality_gate = "PASS" if quality["relative_diff"] is not None and quality["relative_diff"] >= config["quality_relative_floor"] else "FAIL" if quality["relative_diff"] is not None else "NOT_EVALUABLE"
    bias_gate = "PASS" if bias["relative_diff"] is not None and bias["relative_diff"] <= -config["minimum_bias_relative_reduction"] else "FAIL" if bias["relative_diff"] is not None else "NOT_EVALUABLE"
    rng = random.Random(config["bootstrap_seed"])
    paired = [(before[q], after[q]) for q in before]
    intervals = {}
    for key in ("ndcg_at_5", "head_excess"):
        diffs = []
        for _ in range(config["bootstrap_samples"]):
            sample = rng.choices(paired, k=len(paired))
            diffs.append(sum(c[key]-b[key] for b,c in sample) / len(sample))
        diffs.sort()
        intervals[key] = [diffs[int((len(diffs)-1)*.025)], diffs[int((len(diffs)-1)*.975)]]
    quality_evidence = "SUPPORTED" if intervals["ndcg_at_5"][0] >= quality["baseline"]*config["quality_relative_floor"] else "INCONCLUSIVE"
    return {"metrics": values, "paired_users": len(paired), "quality_point_gate": quality_gate,
            "bias_point_gate": bias_gate, "quality_loss_control_evidence": quality_evidence,
            "conditional_user_bootstrap_95": intervals,
            "development_selection": "neutral_user_feedback" if quality_gate == bias_gate == "PASS" else "retain",
            "final_test_promotion": "not_attempted",
            "scope": "Same sampled candidates and trained-state conditional user resampling; not causal debiasing, online A/B, or training-seed uncertainty.",
            "bias_definition": "Mean max(0, top5 head share minus candidate head share); opportunity reference only, not relevance-adjusted ground truth."}


def memory_screen(arm, events, sample_size=5):
    """Flag negative-language traces for reading, never label them causal bias."""
    marker = re.compile(r"\b(dislike\w*|avoid\w*|hate\w*)\b|don.t (?:like|enjoy)|不喜欢|讨厌", re.I)
    flags, examples = 0, []
    for event in events:
        row = json.loads((Path(arm)/(event["event_id"]+".json")).read_text(encoding="utf-8"))
        user = event["user"]
        memory = row["after"]["users"][user]
        marked = bool(marker.search(memory))
        flags += marked
        if len(examples) < sample_size:
            negative = row["before"]["items"][event["negative"]]["title"]
            examples.append({"evidence_id": event["event_id"], "user": user,
                             "observed_title": row["before"]["items"][event["positive"]]["title"],
                             "unobserved_comparison_title": negative,
                             "user_memory_after": memory, "negative_language_marker": marked,
                             "comparison_title_mentioned": negative.lower() in memory.lower()})
    return {"events": len(events), "negative_language_flagged": flags,
            "negative_language_screen_rate": flags/len(events), "cases": examples,
            "meaning": "Lexical screen, not proven aversion or error; 'do not dislike' also flags. Read traces before interpretation."}
