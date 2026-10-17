"""Turn executed comparisons into research evidence; private truth stays in scoring."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from rsi_researcher.evaluation.metrics import evaluate


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records(directory, state):
    return [_read(Path(directory) / (eid + ".json")) for eid in state["completed"]
            if (Path(directory) / (eid + ".json")).exists()]


def _item_note(state, item, owner=None):
    items = state["private_items"][owner] if owner is not None else state["items"]
    return items[item]["memory"]


def _sources(records, kind, key, owner=None):
    return [r["event"]["event_id"] for r in records
            if ((kind == "user" and r["event"]["user"] == key)
                or (kind == "item" and r["event"]["positive"] == key
                    and (owner is None or r["event"]["user"] == owner)))]


def _cross_user_reads(records, private=False):
    """Logged read paths are execution facts, not proof of a causal effect."""
    versions, result = {}, []
    for record in records:
        event = record["event"]
        user, eid = event["user"], event["event_id"]
        for item in event["candidates"]:
            source = versions.get((user if private else None, item))
            if source and source[1] != user:
                result.append({"event_id": eid, "user": user, "item": item,
                               "source_event_id": source[0], "source_user": source[1]})
        versions[(user if private else None, event["positive"])] = (eid, user)
    return result


def build_candidate_evidence(world, baseline_dir, candidate_dir, baseline_pair, candidate_pair, operator):
    """Full JSON-ready evidence for one paired candidate, with fixed denominators.

    Truth utilities enter evaluate() only. Output contains scores, legal notebook
    text, event IDs and execution facts; it never copies hidden world metadata.
    Both rank directions have equal weight. Invalid output retains its penalty.
    """
    if len(baseline_pair) != 2 or len(candidate_pair) != 2:
        raise ValueError("paired evidence requires both forward and reverse rankings")
    queries = world["public"]["queries"]
    private = world["private"]
    paired = []
    for direction, before, after in zip(("forward", "reverse"), baseline_pair, candidate_pair):
        b = evaluate(queries, before, private["utilities"], private["popular"])
        c = evaluate(queries, after, private["utilities"], private["popular"])
        for query, br, cr in zip(queries, b["queries"], c["queries"]):
            paired.append({"query_id": query["query_id"], "user": query["user"], "direction": direction,
                           "baseline_valid": br["valid"], "candidate_valid": cr["valid"],
                           "quality_before": br["ndcg_at_10"], "quality_after": cr["ndcg_at_10"],
                           "quality_change": cr["ndcg_at_10"] - br["ndcg_at_10"],
                           "bias_before": br["bias"], "bias_after": cr["bias"],
                           "bias_change": cr["bias"] - br["bias"],
                           "signed_excess_before": br["signed_excess"], "signed_excess_after": cr["signed_excess"],
                           "baseline_top10": before.get(query["query_id"])[:10] if br["valid"] else None,
                           "candidate_top10": after.get(query["query_id"])[:10] if cr["valid"] else None})
    n = len(paired)
    summary = {"paired_query_directions": n,
               "quality_change_mean": sum(r["quality_change"] for r in paired) / n,
               "bias_change_mean": sum(r["bias_change"] for r in paired) / n,
               "baseline_invalid": sum(not r["baseline_valid"] for r in paired),
               "candidate_invalid": sum(not r["candidate_valid"] for r in paired)}
    for metric in ("quality", "bias"):
        values = [r[metric + "_change"] for r in paired]
        summary[metric + "_increased"] = sum(v > 0 for v in values)
        summary[metric + "_decreased"] = sum(v < 0 for v in values)
        summary[metric + "_unchanged"] = sum(v == 0 for v in values)
        summary[metric + "_change_range"] = [min(values), max(values)]
    summary["quality_up_bias_up"] = sum(r["quality_change"] > 0 and r["bias_change"] > 0 for r in paired)
    summary["quality_down_bias_down"] = sum(r["quality_change"] < 0 and r["bias_change"] < 0 for r in paired)

    baseline_dir, candidate_dir = Path(baseline_dir), Path(candidate_dir)
    bstate = _read(baseline_dir / "state.json")
    # Replay keeps event records at candidate/, and final rankings at candidate/ranking/.
    state_dir = candidate_dir if (candidate_dir / "state.json").exists() else candidate_dir / "ranking"
    cstate = _read(state_dir / "state.json")
    brecords, crecords = _records(baseline_dir, bstate), _records(candidate_dir, cstate)
    source_candidate_records = brecords if operator == "popularity_rerank" else crecords
    changes = []

    def add_change(kind, key, before, after, owner=None):
        if before != after:
            changes.append({"kind": kind, "id": key, "owner": owner,
                            "baseline_note": before, "candidate_note": after,
                            "baseline_source_event_ids": _sources(brecords, kind, key),
                            "candidate_source_event_ids": _sources(source_candidate_records, kind, key, owner)})

    for user in bstate["users"]:
        add_change("user", user, bstate["users"][user], cstate["users"][user])
    if "private_items" in cstate:
        for user, items in cstate["private_items"].items():
            for item in items:
                add_change("item", item, _item_note(bstate, item), _item_note(cstate, item, user), user)
    else:
        for item in bstate["items"]:
            add_change("item", item, _item_note(bstate, item), _item_note(cstate, item))

    bevents = {r["event"]["event_id"]: r["event"] for r in brecords}
    event_changes = [{"event_id": r["event"]["event_id"],
                      "baseline_candidates": bevents[r["event"]["event_id"]]["candidates"],
                      "candidate_candidates": r["event"]["candidates"]}
                     for r in crecords if r["event"]["event_id"] in bevents
                     and r["event"]["candidates"] != bevents[r["event"]["event_id"]]["candidates"]]
    execution = {"baseline_event_count": len(bstate["completed"]),
                 "candidate_completed_event_count": len(cstate["completed"]),
                 "candidate_training_executed": operator != "popularity_rerank",
                 "candidate_event_records_available": len(crecords),
                 "private_item_notebooks": "private_items" in cstate,
                 "candidate_shared_catalog_equals_baseline": cstate["items"] == bstate["items"],
                 "candidate_shared_catalog_equals_initial": cstate["items"] == world["public"]["initial"]["items"],
                 "training_candidate_order_changes": event_changes,
                 "baseline_cross_user_item_reads": _cross_user_reads(brecords),
                 "candidate_cross_user_item_reads": _cross_user_reads(source_candidate_records, "private_items" in cstate),
                 "read_paths_source": "inherited_baseline" if operator == "popularity_rerank" else "candidate_records",
                 "actions": dict(Counter(r.get("action", "learn") for r in crecords))}
    if (candidate_dir / "plan.json").exists():
        plan = _read(candidate_dir / "plan.json")
        execution["replay"] = {"roots": plan["roots"], "affected_events": plan["affected_events"],
                               "reused_events": [r["event"]["event_id"] for r in crecords
                                                 if r.get("action") == "reuse_independent"]}
    return {"operator": operator, "summary": summary, "paired_queries": paired,
            "memory_change_summary": {"changed_user_notes": sum(r["kind"] == "user" for r in changes),
                                      "changed_item_notes": sum(r["kind"] == "item" for r in changes),
                                      "private_item_comparisons": "private_items" in cstate},
            "memory_changes": changes, "execution": execution,
            "interpretation": "Descriptive paired development evidence. Query directions and users are not independent worlds. "
                              "Notebook differences and logged read paths do not establish causes or notebook truth."}


def compact_candidate_evidence(evidence, max_queries=8, max_notes=6, max_events=8, text_limit=600):
    """Bound model input without selecting only successful examples.

    Preserve strongest quality loss and bias increase if present, then fill by
    absolute change. Notes/events retain source order, never a favorable filter.
    """
    if min(max_queries, max_notes, max_events, text_limit) < 1:
        raise ValueError("evidence limits must be positive")
    rows = evidence["paired_queries"]
    chosen = []
    for metric, adverse in (("quality_change", lambda v: v < 0), ("bias_change", lambda v: v > 0)):
        candidates = [i for i, row in enumerate(rows) if adverse(row[metric])]
        if candidates and len(chosen) < max_queries:
            index = max(candidates, key=lambda i: abs(rows[i][metric]))
            if index not in chosen:
                chosen.append(index)
    for index in sorted(range(len(rows)), key=lambda i: -max(abs(rows[i]["quality_change"]), abs(rows[i]["bias_change"]))):
        if len(chosen) == max_queries:
            break
        if index not in chosen:
            chosen.append(index)
    result = {key: deepcopy(evidence[key]) for key in ("operator", "summary", "memory_change_summary", "interpretation")}
    result["paired_queries"] = [deepcopy(rows[i]) for i in chosen]
    result["memory_changes"] = deepcopy(evidence["memory_changes"][:max_notes])
    text_truncated = 0
    for note in result["memory_changes"]:
        for key in ("baseline_note", "candidate_note"):
            if len(note[key]) > text_limit:
                note[key] = note[key][:text_limit] + "… [截取]"
                text_truncated += 1
        for key in ("baseline_source_event_ids", "candidate_source_event_ids"):
            note[key + "_total"] = len(note[key])
            note[key] = note[key][:max_events]
    execution = deepcopy(evidence["execution"])
    event_totals = {}
    for key, value in execution.items():
        if isinstance(value, list):
            event_totals[key] = len(value)
            execution[key] = value[:max_events]
    if "replay" in execution:
        for key, value in execution["replay"].items():
            event_totals["replay." + key] = len(value)
            execution["replay"][key] = value[:max_events]
    result["execution"] = execution
    result["selection"] = {"query_rule": "strongest quality loss and bias increase, then largest absolute changes; stable source-order ties",
                           "paired_queries_total": len(rows), "paired_queries_shown": len(chosen),
                           "memory_rule": "first notes in deterministic user/item order",
                           "memory_changes_total": len(evidence["memory_changes"]),
                           "memory_changes_shown": len(result["memory_changes"]),
                           "text_fields_truncated": text_truncated,
                           "event_rule": "first events in recorded order", "event_list_totals": event_totals}
    return result
