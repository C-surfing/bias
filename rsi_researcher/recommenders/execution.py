"""One bounded development attempt: diagnose, replay, externally score, archive."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from rsi_researcher.runtime.client import API_PROTOCOL, BudgetExceeded, StrictDeepSeek, InvalidOutput, ModelFailure, RANKING_PROTOCOL, MEMORY_CONTROL_PROTOCOL, TEXT_SCHEMA, atomic_json, fields, nonempty_text, object_schema, usage_summary
from rsi_researcher.recommenders.agentcf import AgentCF
from rsi_researcher.paths import ROOT
from rsi_researcher.datasets.controlled import build_world, ordered_events
from rsi_researcher.evaluation.metrics import evaluate, point_decision
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.source_snapshot import snapshot_package


OPERATORS = {
    "retain": "Keep the original feedback prompts unchanged (no-op control).",
    "uncertain_feedback": "Treat a sampled unobserved item as unknown, not disliked; do not invent item attributes.",
}


def intervention(prompts, operator):
    if not isinstance(operator, str) or operator not in OPERATORS:
        raise InvalidOutput("operator is not permitted")
    changed = dict(prompts)
    if operator == "uncertain_feedback":
        context = ("User memory: $user_description\nCandidate descriptions in order:\n$list_of_item_description\n"
                   "Observed interaction: the user consumed '$pos_item_title'. '$neg_item_title' was sampled as an "
                   "unobserved comparison; there is no evidence that the user saw or disliked it. "
                   "The previous system explanation is a prediction, not an observation: $system_reason\n")
        for key in ("user_prompt_template", "user_prompt_template_true"):
            changed[key] = context + ("Update the user's memory using only observed evidence. Preserve uncertainty; "
                                     "do not infer dislikes from missing interactions. Do not invent CD attributes. "
                                     "Keep the memory under 180 words.")
        for key in ("item_prompt_template", "item_prompt_template_true"):
            changed[key] = context + ("Update both descriptions in candidate order, under 50 words each. Preserve "
                                     "known attributes. A user's consumption is not a universal endorsement and "
                                     "does not establish new intrinsic item properties. Preserve unsupported fields unchanged.")
    return changed


def validate_proposal(proposal):
    fields(proposal, ("parent", "operator", "mechanism", "evidence", "expected_direction", "falsifier"))
    if proposal["parent"] != "B0" or not isinstance(proposal["operator"], str) or proposal["operator"] not in OPERATORS:
        raise InvalidOutput("invalid parent or operator")
    for key in ("mechanism", "evidence", "expected_direction", "falsifier"):
        nonempty_text(proposal[key])
    return proposal


def execute_arm(directory, model, prompts, public, events, resume=False, memory_policy="original_update"):
    if memory_policy not in ("original_update", "frozen", "user_only", "item_only", "no_sharing"):
        raise ValueError("unknown memory policy")
    directory.mkdir(exist_ok=resume)
    control = {"memory_policy": memory_policy, "feedback_protocol": MEMORY_CONTROL_PROTOCOL}
    control_path = directory / "memory_control.json"
    if control_path.exists():
        if json.loads(control_path.read_text(encoding="utf-8")) != control:
            raise ValueError("resume memory policy differs")
    elif resume and memory_policy != "original_update":
        raise ValueError("resume memory policy was not recorded")
    else:
        atomic_json(control_path, control)
    state = deepcopy(public["initial"])
    if memory_policy == "no_sharing":
        state["private_items"] = {u: deepcopy(state["items"]) for u in state["users"]}
    kernel = AgentCF(model, prompts)
    for event in events:
        committed = directory / (event["event_id"] + ".json")
        if resume and committed.exists():
            saved = json.loads(committed.read_text(encoding="utf-8"))
            if saved["event"] != event or saved["before"] != state:
                raise ValueError("resume event chain differs from frozen inputs")
            state = saved["after"]
            continue
        view = state
        if memory_policy == "no_sharing":
            view = state | {"items": state["private_items"][event["user"]]}
        after = kernel.event(view, event, "original_update" if memory_policy == "no_sharing" else memory_policy)
        if memory_policy == "no_sharing":
            after["private_items"][event["user"]] = after["items"]
            after["items"] = deepcopy(state["items"])
        atomic_json(committed, {"event": event, "before": state, "after": after, "memory_policy": memory_policy})
        state = after
    atomic_json(directory / "state.json", state)
    if resume and (directory / "rankings.json").exists():
        saved = json.loads((directory / "rankings.json").read_text(encoding="utf-8"))
        return saved["forward"], saved["reverse"]
    rankings, reversed_rankings = {}, {}
    for query in public["queries"]:
        for reverse, destination in ((False, rankings), (True, reversed_rankings)):
            probe = deepcopy(query)
            probe["query_id"] += "-reverse" if reverse else "-forward"
            if reverse:
                probe["candidates"].reverse()
            try:
                view = state | {"items": state["private_items"][query["user"]]} if memory_policy == "no_sharing" else state
                destination[query["query_id"]] = kernel.rank(view, probe)
            except (InvalidOutput, ModelFailure) as exc:
                if isinstance(exc, ModelFailure) and exc.http_status in (401, 402, 403):
                    raise
                destination[query["query_id"]] = None
                atomic_json(directory / (probe["query_id"] + ".failed.json"), {"error": type(exc).__name__})
    atomic_json(directory / "rankings.json", {"forward": rankings, "reverse": reversed_rankings})
    return rankings, reversed_rankings


def score_arm(world, pair):
    private, queries = world["private"], world["public"]["queries"]
    scores = [evaluate(queries, ranking, private["utilities"], private["popular"]) for ranking in pair]
    # Each query and each direction has fixed equal weight, including failures.
    result = {k: sum(s[k] for s in scores) / 2 for k in ("utility", "bias", "signed_excess")}
    result.update(query_count=sum(s["query_count"] for s in scores),
                  invalid_rankings=sum(s["invalid_rankings"] for s in scores))
    result["position_top1_disagreement"] = sum(
        not pair[0][q["query_id"]] or not pair[1][q["query_id"]]
        or pair[0][q["query_id"]][0] != pair[1][q["query_id"]][0] for q in queries) / len(queries)
    return result


def run_attempt(directory, model, world, prompts):
    """Model is an API-only client; no model-generated code or filesystem tools run."""
    directory = Path(directory)
    atomic_json(directory / "world.json", world)
    atomic_json(directory / "protocol.json", {"baseline": "B0", "candidate_limit": 1,
                "ranking_protocol": RANKING_PROTOCOL,
                "api_protocol": API_PROTOCOL,
                "operators": OPERATORS, "minimum_bias": 0.001, "quality_floor_ratio": 0.99,
                "bias_ceiling_ratio": 0.90, "request_limit": model.max_requests,
                "status": "development smoke; coverage/learning guards not calibrated",
                "position": "paired reverse evaluation; paired_0 training", "exposure": "high"})
    atomic_json(directory / "selection.json", {"selected": "B0", "reason": "no accepted candidate yet"})
    events = ordered_events(world["public"]["events"]["high"], world["private"]["popular"], "pair_0")
    try:
        model.scope = {"workflow_arm": "B0"}
        baseline = score_arm(world, execute_arm(directory / "B0", model, prompts, world["public"], events))
        atomic_json(directory / "B0" / "score.json", baseline)
        evidence = {"development_metrics": baseline, "event_count": len(events),
                    "observation": "Positive items were consumed; negatives are sampled unobserved items.",
                    "original_feedback_assumption": "Original prompts describe sampled negatives as disliked.",
                    "allowed_operators": OPERATORS, "parent": "B0"}
        atomic_json(directory / "diagnosis_input.json", evidence)
        model.scope = {"workflow_role": "diagnose_and_propose"}
        proposal = model.ask("Diagnose these development records and choose exactly one allowed operator. "
                             "State a mechanism, evidence, predicted direction, and a falsifiable counterexample. "
                             "Do not claim a causal effect from these small-sample scores.\n" + json.dumps(evidence),
                             object_schema({"parent": {"type": "string", "enum": ["B0"]},
                                            "operator": {"type": "string", "enum": sorted(OPERATORS)},
                                            "mechanism": TEXT_SCHEMA, "evidence": TEXT_SCHEMA,
                                            "expected_direction": TEXT_SCHEMA, "falsifier": TEXT_SCHEMA}),
                             {"stage": "diagnose_and_propose", "workflow": "S1"})
        atomic_json(directory / "proposal.json", proposal)
        validate_proposal(proposal)
        candidate_prompts = intervention(prompts, proposal["operator"])
        atomic_json(directory / "candidate_prompts.json", candidate_prompts)
        model.scope = {"workflow_arm": "candidate"}
        candidate = score_arm(world, execute_arm(directory / "candidate", model, candidate_prompts, world["public"], events))
        atomic_json(directory / "candidate" / "score.json", candidate)
        guards = {"position": candidate["position_top1_disagreement"] <= baseline["position_top1_disagreement"],
                  "budget": model.requests <= model.max_requests, "memory_coverage": None, "learning": None}
        decision = point_decision(baseline, candidate, guards, minimum_bias=0.001)
        atomic_json(directory / "decision.json", decision)
        atomic_json(directory / "selection.json", {"selected": "candidate" if decision["status"] == "eligible" else "B0",
                    "reason": decision["reason"], "statistical_support": "not_tested"})
        return decision
    except (InvalidOutput, ModelFailure, BudgetExceeded) as exc:
        decision = {"status": "failed", "error": type(exc).__name__, "selected": "B0"}
        atomic_json(directory / "decision.json", decision)
        return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--max-requests", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=200000)
    parser.add_argument("--max-seconds", type=int, default=900)
    options = parser.parse_args()
    if min(options.max_requests, options.max_tokens, options.max_seconds) < 1:
        parser.error("limits must be positive")
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot_package(output)
    prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))
    atomic_json(output / "baseline_prompts.json", prompts)
    limits = RunLimits(output, options.max_requests, options.max_tokens, options.max_seconds)
    decision = run_attempt(output, StrictDeepSeek(output, options.max_requests, limits),
                           build_world(options.seed, users=2, events_per_user=1), prompts)
    if (output / "calls.jsonl").exists():
        atomic_json(output / "usage.json", usage_summary(output))
    print(json.dumps(decision))
