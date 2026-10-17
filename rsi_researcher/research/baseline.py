"""Load a fixed development baseline without silently changing the comparison."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from rsi_researcher.runtime.client import API_PROTOCOL, MEMORY_CONTROL_PROTOCOL, MODEL_PARAMETERS, RANKING_PROTOCOL
from rsi_researcher.datasets.controlled import ordered_events
from rsi_researcher.recommenders.memory_dependencies import dependency_graph


UNKNOWN = "unknown"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fingerprint(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def make_baseline_contract(world, prompts, events, metadata):
    """Identify one batch's fixed comparison inputs, without inventing missing settings."""
    keys = ("api_protocol", "ranking_protocol", "memory_control_protocol", "model_parameters")
    recorded = {key: deepcopy(metadata[key]) if metadata.get(key) is not None else UNKNOWN for key in keys}
    contract = {"version": "development_baseline_v1", "world_hash": _fingerprint(world),
                "initial_state_hash": _fingerprint(world["public"]["initial"]),
                "events_hash": _fingerprint(events), "panels_hash": _fingerprint(world["public"]["queries"]),
                "prompts_hash": _fingerprint(prompts), "metadata": recorded,
                "comparison_scope": "development_cached_baseline_not_time_matched",
                "unknown_metadata": [key for key, value in recorded.items() if value == UNKNOWN]}
    contract["fingerprint"] = _fingerprint(contract)
    return contract


def source_prompts(source_arm):
    """Read the prompts that produced this arm, including older pilot layouts."""
    parent = Path(source_arm).resolve().parent
    candidates = ((parent / "baseline_prompts.json", parent / "prompts.json"),
                  (parent.parent / "prompts.json", parent.parent / "input_prompts.json"),
                  (parent.parent.parent / "input_prompts.json",))
    for group in candidates:
        found = []
        for path in group:
            if path.exists():
                value = _read(path)
                if isinstance(value, dict) and "original" in value and isinstance(value["original"], dict):
                    value = value["original"]
                if not isinstance(value, dict) or not value or not all(isinstance(v, str) for v in value.values()):
                    raise ValueError("source baseline prompts are not a prompt dictionary")
                found.append(value)
        if found:
            if any(value != found[0] for value in found[1:]):
                raise ValueError("source baseline prompt records disagree")
            return deepcopy(found[0])
    raise ValueError("source baseline prompts were not recorded")


def _metadata(source_arm, expected_metadata=None):
    """Use recorded execution settings; absent records remain explicitly unknown."""
    source_arm = Path(source_arm)
    expected = {"api_protocol": API_PROTOCOL, "ranking_protocol": RANKING_PROTOCOL,
                "memory_control_protocol": MEMORY_CONTROL_PROTOCOL, "model_parameters": MODEL_PARAMETERS}
    if expected_metadata is not None:
        expected.update({key: value for key, value in expected_metadata.items() if key in expected})
    metadata = {key: UNKNOWN for key in expected}

    def accept(key, value):
        if value is None or value == UNKNOWN:
            return
        if value != expected[key]:
            raise ValueError("source baseline " + key + " differs from the current execution protocol")
        if metadata[key] != UNKNOWN and metadata[key] != value:
            raise ValueError("source baseline execution records disagree")
        metadata[key] = deepcopy(value)

    parent = source_arm.parent
    for directory in (parent, parent.parent, parent.parent.parent):
        for filename in ("protocol.json", "model_config.json", "baseline_contract.json"):
            path = directory / filename
            if not path.exists():
                continue
            recorded = _read(path)
            if filename == "baseline_contract.json":
                recorded = recorded.get("metadata", {})
            for key in expected:
                accept(key, recorded.get(key))
            if filename == "model_config.json":
                accept("model_parameters", recorded.get("parameters"))
        calls = directory / "calls.jsonl"
        if calls.exists():
            with calls.open(encoding="utf-8") as stream:
                for line in stream:
                    call = json.loads(line)
                    if call.get("status") not in ("started", "request"):
                        continue
                    context = call.get("context", {})
                    if context.get("world", parent.name) != parent.name or context.get("research_trial", parent.name) != parent.name:
                        continue
                    if context.get("pilot_arm", source_arm.name) not in ("initial", source_arm.name):
                        continue
                    if context.get("research_arm", "B0") not in ("initial", "B0"):
                        continue
                    accept("api_protocol", call.get("api_protocol"))
                    request = call.get("request", {})
                    model_parameters = expected["model_parameters"]
                    if all(key in request for key in model_parameters):
                        accept("model_parameters", {key: request[key] for key in model_parameters})
                    else:
                        # A partial old record can still reveal an incompatible model.
                        for key in model_parameters.keys() & request.keys():
                            if request[key] != model_parameters[key]:
                                raise ValueError("source baseline recorded model parameter differs: " + key)
    control = source_arm / "memory_control.json"
    if control.exists():
        recorded = _read(control)
        if recorded.get("memory_policy") != "original_update":
            raise ValueError("source baseline is not an original shared-memory update arm")
        accept("memory_control_protocol", recorded.get("feedback_protocol"))
    return metadata


def _ranking_pair(directory, queries):
    saved = _read(Path(directory) / "rankings.json")
    query_ids = [query["query_id"] for query in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("source evaluation contains duplicate query IDs")
    for direction in ("forward", "reverse"):
        rankings = saved[direction]
        if not isinstance(rankings, dict) or set(rankings) != set(query_ids):
            raise ValueError("source rankings do not cover the frozen evaluation queries")
        for query in queries:
            panel, ranking = query["candidates"], rankings[query["query_id"]]
            if len(panel) < 10 or len(set(panel)) != len(panel):
                raise ValueError("source evaluation panel is not valid")
            # None is a recorded ranking failure, retained for fixed-denominator scoring.
            if ranking is not None and (not isinstance(ranking, list) or len(ranking) != len(panel)
                                        or not all(isinstance(item, str) for item in ranking)
                                        or set(ranking) != set(panel)):
                raise ValueError("source ranking does not match the frozen evaluation panel")
    return deepcopy((saved["forward"], saved["reverse"]))


def load_baseline(source_arm, world, prompts, metadata=None):
    """Return a validated snapshot and stable identity; this function never writes."""
    source_arm = Path(source_arm).resolve()
    source_world = _read(source_arm.parent / "world.json")
    if source_world != world:
        raise ValueError("source baseline world differs from the requested world")
    if source_prompts(source_arm) != prompts:
        raise ValueError("source baseline prompts differ from the requested prompts")

    initial = world["public"]["initial"]
    state = _read(source_arm / "state.json")
    completed = state["completed"]
    if not isinstance(completed, list) or len(set(completed)) != len(completed):
        raise ValueError("source baseline completed-event history is invalid")
    records = [_read(source_arm / (event_id + ".json")) for event_id in completed]
    events = [record["event"] for record in records]
    expected_events = ordered_events(world["public"]["events"]["high"], world["private"]["popular"], "pair_0")
    if events != expected_events:
        raise ValueError("source baseline events differ from the full frozen baseline event sequence")
    dependency_graph(records)
    if records:
        if records[0]["before"] != initial or records[-1]["after"] != state:
            raise ValueError("source baseline state does not match its committed event chain")
    elif state != initial:
        raise ValueError("empty baseline event chain differs from initial state")
    initial_dir = source_arm.parent / "initial"
    if _read(initial_dir / "state.json") != initial:
        raise ValueError("source initial arm differs from the frozen initial state")
    queries = world["public"]["queries"]
    initial_pair = _ranking_pair(initial_dir, queries)
    baseline_pair = _ranking_pair(source_arm, queries)
    recorded_metadata = _metadata(source_arm, metadata)
    recorded_contract = source_arm.parent / "baseline_contract.json"
    if recorded_contract.exists():
        saved = _read(recorded_contract)
        # Preserve the settings actually declared when this fixed B0 was established.
        # In particular, an unknown field cannot become evidence through a later copy.
        contract = make_baseline_contract(world, prompts, events, saved["metadata"])
        if saved != contract:
            raise ValueError("source baseline contract differs from the validated comparison inputs")
    else:
        contract = make_baseline_contract(world, prompts, events, recorded_metadata)
    return {"events": deepcopy(events), "records": deepcopy(records), "state": deepcopy(state),
            "initial_pair": initial_pair, "baseline_pair": baseline_pair,
            "contract": contract, "fingerprint": contract["fingerprint"]}
