"""Evidence-linked bias patterns and paired effects on saved AgentCF runs.

No model calls. Patterns describe observable system behavior, not unobserved
exposure probabilities, social influence or true preferences.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from rsi_researcher.datasets.author import popularity_group
from rsi_researcher.evaluation.author import score_author_ranking
from rsi_researcher.evaluation.author_probes import (
    _load_run, _false_dislike, _sentences, probe_conformity_bias, probe_position_bias, kendall_tau,
)

PROTOCOL = "author_pattern_evidence_v1"


def _mean(values):
    return sum(values) / len(values) if values else None


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rank_index(rows):
    result = {r["query"]["query_id"]: r for r in rows}
    if len(result) != len(rows) or not rows:
        raise ValueError("ranking evidence requires unique, nonempty queries")
    return result


def compare_saved_quality(baseline_dir, candidate_dir):
    candidate_dir = Path(candidate_dir)
    if (candidate_dir/"order_controls.json").exists():
        return _read(candidate_dir/"order_controls.json")["quality_comparison"]
    rank_path = lambda directory: (Path(directory)/"ranking.json" if (Path(directory)/"rank_only_config.json").exists()
                                   else Path(directory)/"original_update/ranking.json")
    return paired_quality(_read(rank_path(baseline_dir)), _read(rank_path(candidate_dir)))


def paired_quality(baseline_rows, candidate_rows, samples=2000, seed=20261015, floor_ratio=.99):
    """User-paired conditional uncertainty, including failures in the denominator.

    One author validation query per user. Fixed candidate membership and labels
    make per-query differences comparable. This does not resample training or
    remove dependence induced by shared item memories.
    """
    before, after = _rank_index(baseline_rows), _rank_index(candidate_rows)
    if before.keys() != after.keys():
        raise ValueError("paired quality requires the exact same query cohort")
    pairs = []
    for qid, b in before.items():
        c = after[qid]
        if (b["query"]["user"] != c["query"]["user"]
                or b["query"]["target"] != c["query"]["target"]
                or set(b["query"]["candidates"]) != set(c["query"]["candidates"])):
            raise ValueError("paired quality feedback or candidate membership differs")
        pairs.append((b, c))
    if len({b["query"]["user"] for b, _ in pairs}) != len(pairs):
        raise ValueError("bootstrap unit must be a user; aggregate repeated queries first")
    if samples < 2:
        raise ValueError("at least two bootstrap samples required")
    rng = random.Random(seed)
    margins, changes = [], []
    for _ in range(samples):
        draw = rng.choices(pairs, k=len(pairs))
        changes.append(_mean([c["ndcg_at_5"] - b["ndcg_at_5"] for b, c in draw]))
        margins.append(_mean([c["ndcg_at_5"] - floor_ratio * b["ndcg_at_5"] for b, c in draw]))
    def interval(values):
        values.sort()
        return [values[int((len(values)-1)*.025)], values[int((len(values)-1)*.975)]]
    groups = {}
    for group in ("head", "mid", "tail", "unseen"):
        rows = [(b, c) for b, c in pairs if b["target_group"] == group]
        groups[group] = {"users": len(rows),
                         "baseline_ndcg": _mean([b["ndcg_at_5"] for b, _ in rows]),
                         "candidate_ndcg": _mean([c["ndcg_at_5"] for _, c in rows]),
                         "change": _mean([c["ndcg_at_5"]-b["ndcg_at_5"] for b, c in rows])}
    b_mean = _mean([b["ndcg_at_5"] for b, _ in pairs])
    c_mean = _mean([c["ndcg_at_5"] for _, c in pairs])
    margin_interval = interval(margins)
    valid = all(b.get("valid") and c.get("valid") for b, c in pairs)
    evidence = ("INVALID_EXECUTION" if not valid else "NO_REFERENCE_QUALITY" if b_mean == 0
                else "SUPPORTED_CONDITIONALLY" if margin_interval[0] >= 0
                else "QUALITY_LOSS_SUPPORTED_CONDITIONALLY" if margin_interval[1] < 0
                else "INCONCLUSIVE")
    return {"users": len(pairs), "baseline_ndcg": b_mean, "candidate_ndcg": c_mean,
            "relative_change": (c_mean-b_mean)/b_mean if b_mean else None,
            "point_quality_gate": valid and b_mean > 0 and c_mean >= floor_ratio*b_mean,
            "ndcg_change_95": interval(changes), "quality_margin_95": margin_interval,
            "quality_floor_ratio": floor_ratio, "quality_evidence": evidence,
            "target_groups": groups, "bootstrap_samples": samples, "seed": seed,
            "scope": "Conditional user bootstrap on fixed trained states; not independent validation, training-seed uncertainty or causal debiasing."}


def order_control_effect(original_rows, repeated_rows, reversed_rows):
    """Separate same-order sampling instability from order perturbation effects."""
    original, repeated, reversed_ = map(_rank_index, (original_rows, repeated_rows, reversed_rows))
    if not (original.keys() == repeated.keys() == reversed_.keys()):
        raise ValueError("order controls require the exact same query cohort")
    cases = []
    for qid, first in original.items():
        repeat, reverse = repeated[qid], reversed_[qid]
        query = first["query"]
        if repeat["query"] != query or reverse["query"] != query | {"candidates": list(reversed(query["candidates"]))}:
            raise ValueError("order controls must repeat or reverse only the presentation order")
        valid = all(r.get("valid") for r in (first, repeat, reverse))
        row = {"query_id": qid, "valid": valid}
        if valid:
            noise = (1-kendall_tau(first["ranking"], repeat["ranking"]))/2
            flip = (1-kendall_tau(first["ranking"], reverse["ranking"]))/2
            row.update(same_order_distance=noise, reversed_order_distance=flip,
                       excess_order_instability=flip-noise,
                       top1_same_order_changes=first["ranking"][0] != repeat["ranking"][0],
                       top1_reversed_order_changes=first["ranking"][0] != reverse["ranking"][0])
        cases.append(row)
    valid = [r for r in cases if r["valid"]]
    return {"queries": len(cases), "valid_queries": len(valid),
            "same_order_distance": _mean([r["same_order_distance"] for r in valid]),
            "reversed_order_distance": _mean([r["reversed_order_distance"] for r in valid]),
            "excess_order_instability": _mean([r["excess_order_instability"] for r in valid]),
            "per_query": cases,
            "scope": "Same trained state, prompts and query membership; excess instability subtracts same-order noise. One repeat and one reversal are a screening design, not a model of user click-position bias."}


def describe_run(run_dir, case_limit=5):
    """Build all five pattern cards, exact trace references and subgroup outcomes."""
    run_dir = Path(run_dir).resolve()
    rank_only = (run_dir / "rank_only_config.json").exists()
    training_dir = Path(_read(run_dir/"rank_only_config.json")["source"]) if rank_only else run_dir
    state, records, inputs, profile, rankings, calls = _load_run(training_dir)
    if rank_only:
        rankings = _read(run_dir/"ranking.json")
    counts = profile["training_item_counts"]
    group = lambda item: popularity_group(item, counts, profile["head_min_count"], profile["tail_max_count"])
    source_history_complete="source_events" in inputs or not any("source_event_id" in e for e in inputs["events"])
    source_events = inputs.get("source_events", list({e.get("source_event_id",e["event_id"]):e for e in inputs["events"]}.values()))
    observed_by_user = defaultdict(list)
    for event in source_events:
        observed_by_user[event["user"]].append(event["positive"])
    new_claims, persisted_claims, exposure_cases = 0, 0, []
    last_write, dependencies, update_counts = {}, [], Counter()
    user_training, unique_sources, negative_writes = Counter(), set(), 0
    for record in records:
        e = record["event"]
        user, positive, negative = e["user"], e["positive"], e["negative"]
        title = inputs["initial"]["items"][negative]["title"]
        before, after = record["before"]["users"][user], record["after"]["users"][user]
        flagged=_false_dislike(after,title)
        if flagged:
            existed = _false_dislike(before, title)
            persisted_claims += existed
            new_claims += not existed
            if len(exposure_cases) < case_limit:
                exposure_cases.append({"event_id": e["event_id"], "source_event_id": e.get("source_event_id", e["event_id"]),
                                       "observed_positive_title":inputs["initial"]["items"][positive]["title"],
                                       "comparison_title": title, "already_in_before_memory": existed,
                                       "before": before, "after": after,
                                       "assertion_quote":next(s for s in _sentences(after) if _false_dislike(s,title)),
                                       "trace": str(training_dir/"original_update"/(e["event_id"]+".json"))})
        for item in e["candidates"]:
            source = last_write.get(item)
            if source and source["user"] != user and record["before"]["items"][item]["memory"] == source["memory"]:
                dependencies.append({"item": item, "writer_event": source["event_id"],
                                     "reader_event": e["event_id"], "writer_user": source["user"], "reader_user": user,
                                     "writer_event_had_dislike_cue":source["had_dislike_cue"]})
        if record["after"]["items"][positive]["memory"] != record["before"]["items"][positive]["memory"]:
            last_write[positive] = {"event_id": e["event_id"], "user": user, "memory": record["after"]["items"][positive]["memory"],
                                    "had_dislike_cue":flagged}
        negative_writes += record["after"]["items"][negative]["memory"] != record["before"]["items"][negative]["memory"]
        update_counts[group(positive)] += 1
        user_training[user] += 1
        unique_sources.add(e.get("source_event_id", e["event_id"]))
    measured = []
    for row in rankings:
        query = row["query"]
        score = score_author_ranking(row["ranking"] if row.get("valid") else [], query, profile)
        history = observed_by_user[query["user"]]
        history_head = _mean([int(group(i) == "head") for i in history])
        measured.append({"query_id": query["query_id"], "user": query["user"], "valid": row.get("valid", False),
                         **score, "observed_history_head_share": history_head,
                         "popularity_calibration_gap": abs(score["top5_head_share"]-history_head) if history_head is not None else None})
    target_groups = {}
    for label in ("head", "mid", "tail", "unseen"):
        rows = [r for r in measured if r["target_group"] == label]
        target_groups[label] = {"queries": len(rows), "ndcg_at_5": _mean([r["ndcg_at_5"] for r in rows]),
                                "hit_at_5": _mean([r["hit_at_5"] for r in rows])}
    conf = probe_conformity_bias(state, calls)
    position = probe_position_bias(rankings)
    cards = {
        "exposure": {"pattern": "sampled comparison becomes an asserted dislike",
                     "new_explicit_claim_events": new_claims, "persisted_explicit_claim_events": persisted_claims,
                     "training_events": len(records), "cases": exposure_cases,
                     "limit": "Explicit-text screen; missing exposure is unknown, not known non-exposure. A mention may refer to a different clause and needs human review."},
        "selection": {"pattern": "learning opportunity and observed support imbalance",
                      "scheduled_events": len(records), "unique_source_events_used": len(unique_sources),
                      "original_observed_events": len(source_events), "scheduled_group_events": dict(update_counts),
                      "source_history_complete":source_history_complete,
                      "original_group_events": dict(Counter(group(e["positive"]) for e in source_events)),
                      "training_users": len(user_training), "per_user_training_events": dict(user_training),
                      "limit": "Replay weights do not create independent feedback or identify selection propensities."},
        "conformity": {"pattern": "reputation/popularity vocabulary in preference reasoning",
                       "reason_lexicon_rate": conf["select_reason_popularity_rate"],
                       "limit": "Vocabulary is not proof of social conformity; matched cue perturbations are still needed."},
        "position": {"pattern": "ranking depends on candidate presentation order",
                     "descriptive_input_order_tau": position["input_order_following_kendall_tau"],
                     "control_status": "available" if (run_dir/"order_controls.json").exists() else "not_run",
                     "limit": "A single ordering cannot separate order effects from sampling noise."},
        "popularity": {"pattern": "head-item exposure exceeds the candidate opportunity mix",
                       "mean_signed_excess": _mean([r["excess_popularity_exposure"] for r in measured]),
                       "mean_positive_excess": _mean([max(0,r["excess_popularity_exposure"]) for r in measured]),
                       "mean_history_calibration_gap": _mean([r["popularity_calibration_gap"] for r in measured if r["popularity_calibration_gap"] is not None]),
                       "limit": "Candidate mix and observed taste are references, not full relevance labels; less popularity is not automatically better."},
    }
    if cards["position"]["control_status"] == "available":
        cards["position"]["controls"] = _read(run_dir/"order_controls.json")
    return {"protocol": PROTOCOL, "dataset": profile.get("dataset"),
            "source": str(run_dir), "training_source": str(training_dir), "rank_only": rank_only,
            "patterns": cards, "target_group_quality": target_groups, "per_query": measured,
            "shared_memory": {"observed_cross_user_reads": len(dependencies),
                              "reads_after_flagged_writer_events":sum(e["writer_event_had_dislike_cue"] for e in dependencies),
                              "examples": dependencies[:case_limit], "negative_item_memory_writes": negative_writes,
                              "interpretation": "Exact read-after-write dependencies show possible propagation routes, not causal attribution of a quality or bias change."},
            "evidence_level": "observational_pattern_screen"}


def render_patterns(report):
    p = report["patterns"]
    lines = ["# AgentCF 五类偏差模式与证据", "", f"数据：{report['dataset']}；证据版本：{report['protocol']}。",
             "只分析已保存的真实记录，不调用模型。五类模式不等于五类真实偏差均已识别或消除。", "",
             "| 方向 | 实际观察 | 还需验证 |", "|---|---|---|",
             f"| 曝光相关 | 新出现显式负面表述 {p['exposure']['new_explicit_claim_events']}，既有表述 {p['exposure']['persisted_explicit_claim_events']} | 逐句检查指代与依据，不能把文字命中率叫真实错误率 |",
             f"| 选择相关 | {p['selection']['scheduled_events']} 次学习来自 {p['selection']['unique_source_events_used']} 个不同事件 | 更多更新不保证内容可信，重复记录不算新证据 |",
             f"| 从众相关 | 理由热门用词率 {p['conformity']['reason_lexicon_rate']} | 做同一内容的名气线索配对对照 |",
             f"| 位置相关 | 输入顺序相关 {p['position']['descriptive_input_order_tau']}；配对对照 {p['position']['control_status']} | 与同序重复的自然波动比较 |",
             f"| 热门相关 | 平均额外热门占比 {p['popularity']['mean_signed_excess']} | 同时看各组质量及用户历史偏好 |", "",
             "## 已知目标的分组质量", "", "| 训练热门组 | 验证题数 | NDCG@5 | Hit@5 |", "|---|---:|---:|---:|"]
    for group, row in report["target_group_quality"].items():
        lines.append(f"| {group} | {row['queries']} | {row['ndcg_at_5']} | {row['hit_at_5']} |")
    lines += ["", f"共享物品记忆出现 {report['shared_memory']['observed_cross_user_reads']} 次跨用户再次读取，"
              f"其中 {report['shared_memory']['reads_after_flagged_writer_events']} 次读取的起点事件同时有负面表述标记。"
              "这是可能的传播接触路径，不证明负面推断确实进入了物品记忆或改变了后续结果。", "",
              "## 需要人工看原文的负面表述", ""]
    for case in p["exposure"]["cases"]:
        lines += [f"- `{case['event_id']}`：观察到消费的是 **{case['observed_positive_title']}**；抽样比较的是 **{case['comparison_title']}**。",
                  f"  - 对照标题的负面表述之前已有：{case['already_in_before_memory']}。程序没有该对照 CD 的真实反感反馈，需要核查这句话的依据和指代。",
                  f"  - 命中原句：{case['assertion_quote']}", f"  - [完整前后记忆与原始记录]({Path(case['trace']).as_posix()})"]
    return "\n".join(lines)+"\n"


def pattern_brief(report):
    """Bounded facts for the small researcher model; full traces stay on disk."""
    p = report["patterns"]
    return {
        "protocol": report["protocol"],
        "exposure": {"new_explicit_claims": p["exposure"]["new_explicit_claim_events"],
                     "persisted_explicit_claims": p["exposure"]["persisted_explicit_claim_events"],
                     "training_events": p["exposure"]["training_events"],
                     "examples": [{"event_id": c["event_id"], "comparison_title": c["comparison_title"],
                                   "already_in_memory": c["already_in_before_memory"],
                                   "observed_positive_title":c["observed_positive_title"],
                                   "current_memory_quote":c["assertion_quote"][:500]}
                                  for c in p["exposure"]["cases"][:2]]},
        "selection": {k:p["selection"][k] for k in ("scheduled_events", "unique_source_events_used", "original_observed_events", "scheduled_group_events")},
        "conformity": {"reason_lexicon_rate":p["conformity"]["reason_lexicon_rate"], "evidence":"vocabulary proxy only"},
        "position": {"control_status":p["position"]["control_status"], "need":"same-state same-order-repeat and reversed-order comparison"},
        "popularity": {k:p["popularity"][k] for k in ("mean_signed_excess","mean_history_calibration_gap")},
        "target_group_quality": report["target_group_quality"],
        "cross_user_read_after_write_count": report["shared_memory"]["observed_cross_user_reads"],
        "reads_after_flagged_writer_events":report["shared_memory"]["reads_after_flagged_writer_events"],
        "scope": "Current-run pattern screens; no causal claim. Update counts are action compliance, not evidence quality.",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = describe_run(args.source)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"patterns.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (directory/"report.md").write_text(render_patterns(report),encoding="utf-8")
    print(json.dumps({"status":"completed","model_calls":0,"output":str(directory.resolve())}))


if __name__ == "__main__":
    main()
