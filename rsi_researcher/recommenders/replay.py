"""Repair specified source events and replay their read-dependent descendants."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from rsi_researcher.runtime.client import atomic_json, usage_summary
from rsi_researcher.runtime.models import create_model, model_description
from rsi_researcher.recommenders.agentcf import AgentCF
from rsi_researcher.paths import ROOT
from rsi_researcher.recommenders.execution import intervention
from rsi_researcher.recommenders.memory_dependencies import affected_events, dependency_graph
from rsi_researcher.runtime.source_snapshot import snapshot_package, verify_snapshot


def repair(records, roots, original_kernel, repaired_kernel, output):
    graph = dependency_graph(records)
    affected = affected_events(graph, roots)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "plan.json", {"roots": list(roots), "affected_events": affected,
                "rule": "changed feedback at roots only; original rule for descendants",
                "source_graph": graph})
    atomic_json(directory / "source_records.json", records)
    state = deepcopy(records[0]["before"])
    rebuilt = []
    try:
        for record in records:
            event = record["event"]
            eid = event["event_id"]
            before = deepcopy(state)
            if eid in affected:
                kernel = repaired_kernel if eid in roots else original_kernel
                state = kernel.event(deepcopy(state), deepcopy(event), "original_update")
                action = "repair_root" if eid in roots else "replay_descendant"
            else:
                # Copy only independent writes, never restore an old full-state snapshot.
                for read in graph[eid]["reads"]:
                    if state[read["kind"]][read["id"]] != record["before"][read["kind"]][read["id"]]:
                        raise ValueError("cached event read changed; dependency plan is incomplete")
                for write in graph[eid]["writes"]:
                    state[write["kind"]][write["id"]] = deepcopy(record["after"][write["kind"]][write["id"]])
                state["completed"].append(eid)
                action = "reuse_independent"
            rebuilt.append({"event": deepcopy(event), "before": before, "after": deepcopy(state), "action": action})
            atomic_json(directory / (eid + ".json"), rebuilt[-1])
        dependency_graph(rebuilt)
        atomic_json(directory / "state.json", state)
        atomic_json(directory / "result.json", {"status": "completed", "reexecuted": affected,
                    "reused": [eid for eid in graph if eid not in affected],
                    "promotion": "not_evaluated"})
        return state
    except Exception as exc:
        atomic_json(directory / "result.json", {"status": "failed", "error_type": type(exc).__name__,
                    "committed_final_state": False, "promotion": "not_evaluated"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, help="normal-update source arm, not a frozen-memory arm")
    parser.add_argument("--root-event", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-requests", type=int, default=30)
    parser.add_argument("--config", default=str(ROOT / "configs/qwen_agentcf_author.json"))
    args = parser.parse_args()
    if args.max_requests < 1:
        parser.error("request budget must be positive")
    arm = Path(args.arm)
    source_state = json.loads((arm / "state.json").read_text(encoding="utf-8"))
    records = [json.loads((arm / (eid + ".json")).read_text(encoding="utf-8")) for eid in source_state["completed"]]
    if not records or records[-1]["after"] != source_state:
        raise ValueError("source has no complete matching event chain")
    if arm.name not in ("original_update", "B0"):
        raise ValueError("only original-update baseline arms are supported")
    source_run = next((parent for parent in arm.resolve().parents
                       if (parent / "source_manifest.json").is_file()
                       and (parent / "code/rsi_researcher").is_dir()), None)
    if source_run is None:
        raise ValueError("source has no package snapshot; historical flat runs must use their original saved code")
    verify_snapshot(source_run)
    if (arm.parent / "baseline_prompts.json").exists():
        prompts = json.loads((arm.parent / "baseline_prompts.json").read_text(encoding="utf-8"))
    else:
        prompts = json.loads((arm.parent / "input.json").read_text(encoding="utf-8"))["prompts"]
    output = Path(args.output)
    if output.exists():
        raise ValueError("repair output already exists; source and prior branches must stay unchanged")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model = create_model(output, args.max_requests, config=config)
    try:
        repair(records, [args.root_event], AgentCF(model, prompts),
               AgentCF(model, intervention(prompts, "uncertain_feedback")), output)
    finally:
        if output.exists():
            atomic_json(output / "run_config.json", {"source_arm": str(arm.resolve()),
                        "root_event": args.root_event, "max_requests": args.max_requests,
                        "operator": "uncertain_feedback", "api_protocol": model_description(config)["api_protocol"],
                        "model_config": model_description(config)})
            if (output / "calls.jsonl").exists():
                atomic_json(output / "usage.json", usage_summary(output))
            snapshot_package(output)
            atomic_json(output / "prompts.json", {"original": prompts, "repair": intervention(prompts, "uncertain_feedback")})
