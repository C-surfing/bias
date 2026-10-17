"""Fresh paired-order evaluation of initial, original, and repaired author states."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from rsi_researcher.runtime.client import API_PROTOCOL, RANKING_PROTOCOL, StrictDeepSeek, InvalidOutput, ModelFailure, atomic_json, usage_summary
from rsi_researcher.recommenders.agentcf import AgentCF
from rsi_researcher.paths import ROOT
from rsi_researcher.diagnostics.agentcf_smoke import ndcg
from rsi_researcher.recommenders.memory_dependencies import dependency_graph
from rsi_researcher.runtime.source_snapshot import snapshot_package


def position_distance(first, second):
    if not first or not second:
        return 1.0
    if len(first) < 2 or len(first) != len(second) or len(set(first)) != len(first) or set(first) != set(second):
        raise ValueError("position comparison needs equal complete panels")
    locations = {item: rank for rank, item in enumerate(second)}
    inversions = sum(locations[first[i]] > locations[first[j]] for i in range(len(first)) for j in range(i + 1, len(first)))
    return inversions / (len(first) * (len(first) - 1) / 2)


def evaluate_states(model, prompts, states, queries, directory):
    if not queries or len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("need distinct evaluation queries")
    for q in queries:
        if len(q["candidates"]) < 2 or len(set(q["candidates"])) != len(q["candidates"]) or q["target"] not in q["candidates"]:
            raise ValueError("invalid candidate panel or target")
    directory = Path(directory)
    kernel = AgentCF(model, prompts)
    rows = {arm: [] for arm in states}
    # Interleave arms by query, so an arm is not evaluated only at the end of a batch.
    for query in queries:
        for arm, state in states.items():
            pair = []
            for reverse in (False, True):
                probe = {"query_id": f"{arm}-{query['query_id']}-{'reverse' if reverse else 'forward'}",
                         "user": query["user"], "candidates": list(reversed(query["candidates"])) if reverse else list(query["candidates"])}
                try:
                    ranking = kernel.rank(state, probe)
                    result = {"ranking": ranking, "ndcg_at_10": ndcg(ranking, query["target"]), "valid": True}
                except (InvalidOutput, ModelFailure) as exc:
                    if isinstance(exc, ModelFailure) and exc.http_status in (401, 402, 403):
                        raise
                    result = {"ranking": [], "ndcg_at_10": 0.0, "valid": False, "error_type": type(exc).__name__}
                pair.append(result)
            rows[arm].append({"query_id": query["query_id"], "pair": pair,
                              "top1_disagreement": not pair[0]["ranking"] or not pair[1]["ranking"] or pair[0]["ranking"][0] != pair[1]["ranking"][0],
                              "rank_distance": position_distance(pair[0]["ranking"], pair[1]["ranking"])})
            atomic_json(directory / (arm + ".rankings.json"), rows[arm])
    summary = {}
    for arm, entries in rows.items():
        summary[arm] = {"query_count": len(entries), "ranking_count": len(entries) * 2,
                        "utility": sum(p["ndcg_at_10"] for row in entries for p in row["pair"]) / (2 * len(entries)),
                        "invalid_rankings": sum(not p["valid"] for row in entries for p in row["pair"]),
                        "top1_disagreement": sum(row["top1_disagreement"] for row in entries) / len(entries),
                        "rank_distance": sum(row["rank_distance"] for row in entries) / len(entries)}
    atomic_json(directory / "summary.json", {"arms": summary,
                "purpose": "author-log descriptive check; no hidden-utility bias metric or scientific promotion"})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--repair", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-requests", type=int, default=18)
    args = parser.parse_args()
    if args.max_requests < 1:
        parser.error("budget must be positive")
    source, branch, output = Path(args.source), Path(args.repair), Path(args.output)
    inputs = json.loads((source / "input.json").read_text(encoding="utf-8"))
    original = json.loads((source / "original_update/state.json").read_text(encoding="utf-8"))
    repaired = json.loads((branch / "state.json").read_text(encoding="utf-8"))
    if json.loads((branch / "result.json").read_text(encoding="utf-8"))["status"] != "completed":
        raise ValueError("repair is incomplete")
    source_records = [json.loads((source / "original_update" / (e + ".json")).read_text(encoding="utf-8")) for e in original["completed"]]
    if json.loads((branch / "source_records.json").read_text(encoding="utf-8")) != source_records:
        raise ValueError("repair came from another source")
    for arm_path, state in ((source / "original_update", original), (branch, repaired)):
        records = [json.loads((arm_path / (e + ".json")).read_text(encoding="utf-8")) for e in state["completed"]]
        dependency_graph(records)
        if [r["event"] for r in records] != inputs["events"] or records[0]["before"] != inputs["initial"] or records[-1]["after"] != state:
            raise ValueError("state or event facts do not match frozen inputs")
    output.mkdir(parents=True, exist_ok=False)
    states = {"initial": inputs["initial"], "original": original, "repaired": repaired}
    atomic_json(output / "input.json", {"states": states, "queries": inputs["queries"], "prompts": inputs["prompts"], "max_requests": args.max_requests,
                                       "api_protocol": API_PROTOCOL, "ranking_protocol": RANKING_PROTOCOL})
    snapshot_package(output)
    try:
        result = evaluate_states(StrictDeepSeek(output, args.max_requests), inputs["prompts"], states, inputs["queries"], output)
        print(json.dumps(result))
    except Exception as exc:
        atomic_json(output / "failure.json", {"error_type": type(exc).__name__})
        raise
    finally:
        if (output / "calls.jsonl").exists():
            atomic_json(output / "usage.json", usage_summary(output))
