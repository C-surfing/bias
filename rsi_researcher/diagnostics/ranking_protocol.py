"""Fresh ranking-only check on frozen states, with identical output rules in both arms."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from rsi_researcher.runtime.client import API_PROTOCOL, StrictDeepSeek, RANKING_PROTOCOL, atomic_json, usage_summary
from rsi_researcher.paths import ROOT
from rsi_researcher.recommenders.execution import execute_arm, score_arm
from rsi_researcher.recommenders.memory_dependencies import dependency_graph
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.source_snapshot import snapshot_package


def run_check(output, model, world, states, prompts, resume=False):
    results = {}
    for arm, state in states.items():
        evaluation_world = deepcopy(world)
        evaluation_world["public"]["initial"] = deepcopy(state)
        for query in evaluation_world["public"]["queries"]:
            query["query_id"] = arm + "-" + query["query_id"]
        pair = execute_arm(output / arm, model, prompts, evaluation_world["public"], [], resume=resume)
        results[arm] = score_arm(evaluation_world, pair)
        atomic_json(output / arm / "score.json", results[arm])
    atomic_json(output / "summary.json", {"ranking_protocol": RANKING_PROTOCOL, "api_protocol": API_PROTOCOL, "arms": results,
                "purpose": "fresh outputs on fixed development states; no training, no statistical promotion"})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    world_dir = source / args.world
    world = json.loads((world_dir / "world.json").read_text(encoding="utf-8"))
    learned = json.loads((world_dir / "original_update/state.json").read_text(encoding="utf-8"))
    records = [json.loads((world_dir / "original_update" / (eid + ".json")).read_text(encoding="utf-8")) for eid in learned["completed"]]
    dependency_graph(records)
    if not records or records[0]["before"] != world["public"]["initial"] or records[-1]["after"] != learned:
        raise ValueError("frozen state does not match source event chain")
    prompts = json.loads((source / "prompts.json").read_text(encoding="utf-8"))
    states = {"initial": world["public"]["initial"], "original_update": learned}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "input.json", {"world": world, "states": states, "prompts": prompts,
                "ranking_protocol": RANKING_PROTOCOL, "api_protocol": API_PROTOCOL, "source": str(world_dir.resolve())})
    snapshot_package(output)
    limits = RunLimits(output, 96, 400000, 600)
    try:
        run_check(output, StrictDeepSeek(output, 96, limits), world, states, prompts)
    except Exception as exc:
        atomic_json(output / "failure.json", {"error_type": type(exc).__name__})
        raise
    finally:
        if (output / "calls.jsonl").exists():
            atomic_json(output / "usage.json", usage_summary(output))
            log = [json.loads(line) for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
            atomic_json(output / "format_counts.json", {status: sum(r["status"] == status for r in log)
                        for status in ("ranking_membership_invalid", "ranking_correction_succeeded", "ranking_correction_failed")})
