"""Reconstruct conservative event dependencies from committed AgentCF records."""

import argparse
import json
from pathlib import Path

from rsi_researcher.runtime.client import atomic_json


def dependency_graph(records):
    versions, graph, previous = {}, {}, None
    for record in records:
        if record.get("memory_policy", "original_update") != "original_update" or "private_items" in record["before"]:
            raise ValueError("dependency reconstruction requires normal shared-memory updates")
        event, before, after = record["event"], record["before"], record["after"]
        eid, user, positive = event["event_id"], event["user"], event["positive"]
        if eid in graph or (previous is not None and before != previous):
            raise ValueError("duplicate event or broken state chain")
        if eid in before["completed"] or after["completed"] != before["completed"] + [eid]:
            raise ValueError("event commit does not match state history")
        if before["completed"] != list(graph):
            raise ValueError("dependency reconstruction requires the full initial event chain")
        candidates = event["candidates"]
        if len(candidates) != 2 or set(candidates) != {positive, event["negative"]}:
            raise ValueError("invalid event panel")
        reads = [("users", user)] + [("items", item) for item in candidates]
        writes = [("users", user), ("items", positive)]
        for section in ("users", "items"):
            if before[section].keys() != after[section].keys():
                raise ValueError("state membership changed")
            for key in before[section]:
                if (section, key) not in writes and before[section][key] != after[section][key]:
                    raise ValueError("unexpected write outside event footprint")
        graph[eid] = {
            "reads": [{"kind": section, "id": key, "source_event": versions.get((section, key))}
                      for section, key in reads],
            "parents": sorted({versions[r] for r in reads if r in versions}),
            "writes": [{"kind": section, "id": key} for section, key in writes],
        }
        # Normal AgentCF writes user and positive item each event, even for unchanged text.
        for key in writes:
            versions[key] = eid
        previous = after
    return graph


def affected_events(graph, roots):
    if not roots or not set(roots) <= graph.keys():
        raise ValueError("repair roots must be existing event IDs")
    affected = set(roots)
    for eid, record in graph.items():
        if affected.intersection(record["parents"]):
            affected.add(eid)
    return [eid for eid in graph if eid in affected]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--root-event", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    arm = Path(options.arm)
    state = json.loads((arm / "state.json").read_text(encoding="utf-8"))
    records = [json.loads((arm / (eid + ".json")).read_text(encoding="utf-8")) for eid in state["completed"]]
    graph = dependency_graph(records)
    if records and records[-1]["after"] != state:
        raise ValueError("current state does not match committed chain")
    output = Path(options.output)
    if output.exists():
        raise ValueError("output already exists")
    atomic_json(output, {"graph": graph, "root": options.root_event,
                        "affected_events": affected_events(graph, [options.root_event]),
                        "meaning": "conservative read dependencies, not causal proof; no repair executed"})
