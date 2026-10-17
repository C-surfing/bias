"""Small, failure-conditioned check of DeepSeek Beta strict result submission."""

import argparse
import json
from pathlib import Path

from rsi_researcher.runtime.client import StrictDeepSeek, InvalidOutput, atomic_json, fields, nonempty_text, usage_summary
from rsi_researcher.paths import ROOT
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.source_snapshot import snapshot_package


def failed_cases(source):
    rows = [json.loads(line) for line in (source / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    starts = {row["call_id"]: row for row in rows if row["status"] == "started"}
    cases = {}
    for row in rows:
        if row["status"] != "invalid_response":
            continue
        start = starts[row["call_id"]]
        if row.get("reason") == "empty_content" and start["attempt"] < 3:
            continue
        context = start["context"]
        stage = context["stage"]
        text = {"type": "string", "pattern": "\\S"}
        if stage == "user_update":
            properties = {"memory": text}
        elif stage == "select":
            world_path = (source / context["world"] / "world.json").resolve()
            if source.resolve() not in world_path.parents:
                raise ValueError("source world path is outside the run")
            world = json.loads(world_path.read_text(encoding="utf-8"))
            event = next(e for e in world["public"]["events"]["high"] if e["event_id"] == context["event"])
            properties = {"selected_id": {"type": "string", "enum": sorted(event["candidates"])}, "reason": text}
        else:
            raise ValueError("this small probe only covers the observed select/user failures")
        schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
        cases[json.dumps(context, sort_keys=True)] = {"source_call": row["call_id"], "context": context,
                    "prompt": start["request"]["messages"][1]["content"], "schema": schema}
    return list(cases.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    cases = failed_cases(source)
    if not 1 <= len(cases) <= 3:
        raise ValueError("probe is bounded to at most three known failed inputs")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "input.json", {"cases": cases, "purpose": "known-failure format preflight; not scientific evaluation"})
    snapshot_package(output)
    limits = RunLimits(output, 9, 100000, 300)
    model = StrictDeepSeek(output, 9, limits)
    results = []
    try:
        for case in cases:
            result = model.ask(case["prompt"], case["schema"], case["context"] | {"probe": "strict_output"})
            fields(result, case["schema"]["required"])
            for name, value in result.items():
                nonempty_text(value)
                allowed = case["schema"]["properties"][name].get("enum")
                if allowed is not None and value not in allowed:
                    raise InvalidOutput("result outside permitted IDs")
            results.append({"source_call": case["source_call"], "valid": True, "result": result})
            atomic_json(output / "results.json", results)
        atomic_json(output / "summary.json", {"cases": len(cases), "valid": len(results),
                    "source_run_modified": False, "scientific_support": "not_tested"})
    except Exception as exc:
        atomic_json(output / "failure.json", {"error_type": type(exc).__name__, "completed_cases": len(results)})
        raise
    finally:
        if (output / "calls.jsonl").exists():
            atomic_json(output / "usage.json", usage_summary(output))
