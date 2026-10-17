"""One correction request per saved invalid ranking; engineering diagnosis only."""

import argparse
import json
from pathlib import Path

from rsi_researcher.recommenders.agentcf import AgentCF
from rsi_researcher.runtime.client import StrictDeepSeek, InvalidOutput, ModelFailure, atomic_json, logged_response_content, parse_json, usage_summary, repair_ranking_membership as correct_format
from rsi_researcher.paths import ROOT
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.source_snapshot import snapshot_package


def collect_cases(source, world_id):
    world_dir = source / world_id
    world = json.loads((world_dir / "world.json").read_text(encoding="utf-8"))
    state = json.loads((world_dir / "original_update/state.json").read_text(encoding="utf-8"))
    saved = json.loads((world_dir / "original_update/rankings.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (source / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    started = {r["call_id"]: r for r in records if r["status"] == "started"}
    responses = {}
    for row in records:
        if row["status"] != "received":
            continue
        context = started[row["call_id"]]["context"]
        if context.get("world") == world_id and context.get("pilot_arm") == "original_update" and context.get("stage") == "rank":
            responses[context["query"]] = row
    cases = []
    for query in world["public"]["queries"]:
        for direction in ("forward", "reverse"):
            if saved[direction][query["query_id"]] is not None:
                continue
            qid = query["query_id"] + "-" + direction
            response = responses[qid]
            previous = parse_json(logged_response_content(response))
            panel = list(reversed(query["candidates"])) if direction == "reverse" else query["candidates"]
            cases.append({"source_call": response["call_id"], "query": {"query_id": qid, "user": query["user"], "candidates": panel}, "previous": previous})
    return state, cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    state, cases = collect_cases(source, args.world)
    output.mkdir(parents=True, exist_ok=False)
    prompts = json.loads((source / "prompts.json").read_text(encoding="utf-8"))
    atomic_json(output / "input.json", {"state": state, "cases": cases, "prompts": prompts,
                "purpose": "failure-conditioned format diagnostic; not an unbiased effect comparison"})
    snapshot_package(output)
    limits = RunLimits(output, 18, 200000, 600)
    model = StrictDeepSeek(output, 18, limits)
    kernel, results = AgentCF(model, prompts), []
    try:
        for case in cases:
            try:
                ranking = correct_format(kernel, state, case["query"], case["previous"])
                results.append({"source_call": case["source_call"], "valid": True, "ranking": ranking})
            except (InvalidOutput, ModelFailure) as exc:
                if isinstance(exc, ModelFailure) and exc.http_status in (401, 402, 403):
                    raise
                results.append({"source_call": case["source_call"], "valid": False, "error_type": type(exc).__name__})
            atomic_json(output / "results.json", results)
        atomic_json(output / "summary.json", {"cases": len(cases), "valid_corrections": sum(r["valid"] for r in results),
                    "original_pilot_modified": False, "research_promotion": "not_tested"})
    finally:
        if (output / "calls.jsonl").exists():
            atomic_json(output / "usage.json", usage_summary(output))
