"""Select agent-proposed experiments by predictive disagreement AND executability.

This is a small planning layer, not a new Bias taxonomy or a causal oracle.
An unrestricted scientific concept may require an unsupported intervention.
Only a bounded development-only SQL observation can run through this gateway;
all other capabilities are left for their dedicated, identified executors.
"""
from __future__ import annotations

from rsi_researcher.research.discovery.research_handoff import previously_measured
from rsi_researcher.research.discovery.variable_lineage import (
    audit_construct_overlap, audit_observation_sql,
)

DIRECTIONS = {"increase": 1, "no_change": 0, "decrease": -1}
CAPABILITY_SCHEMA = (
    "sql_observation", "user_contrast", "randomized_exposure",
    "dense_relevance", "controlled_parameter",
    "sequential_intervention", "policy_comparison",
)
CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array", "minItems": 2, "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "concept_id": {"type": "string"},
                    "question": {"type": "string"},
                    "required_capability": {
                        "type": "string", "enum": list(CAPABILITY_SCHEMA)},
                    "study_unit": {"type": "string"},
                    "target_quantity": {"type": "string"},
                    "prediction_h1": {"type": "string"},
                    "prediction_h2": {"type": "string"},
                    "direction_h1": {"type": "string",
                                     "enum": list(DIRECTIONS)},
                    "direction_h2": {"type": "string",
                                     "enum": list(DIRECTIONS)},
                    "sql": {"type": "string"},
                    "reference_assumption": {"type": "string"},
                    "mechanism_link": {"type": "string"},
                    # Optional in old fixtures; new Qwen designs should name
                    # their actual recommendation decision and reference.
                    "decision_target": {"type": "string", "enum": [
                        "exposure_allocation", "ranking", "candidate_generation",
                        "model_state", "utility", "behavior_only", "unknown"]},
                    "reference_kind": {"type": "string", "enum": [
                        "none", "random_insertions", "dense_relevance",
                        "owned_model", "other"]},
                    "predictor_column": {"type": "string"},
                    "outcome_column": {"type": "string"},
                },
                "required": [
                    "concept_id", "question", "required_capability",
                    "study_unit", "target_quantity", "prediction_h1",
                    "prediction_h2", "direction_h1", "direction_h2",
                    "sql", "reference_assumption", "mechanism_link",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["candidates"], "additionalProperties": False,
}


def available_capabilities(adapter) -> dict:
    """Capabilities of real executors, not permission to infer causality."""
    caps = adapter.capabilities
    return {
        "sql_observation": {"available": True,
                            "scope": "development_descriptive_only"},
        "user_contrast": {"available": bool(caps.get("observed_responses")),
                          "scope": "development_observational_user_statistics"},
        "randomized_exposure": {
            "available": bool(caps.get("randomized_exposure_probe")),
            "scope": "restricted_random_insertion_not_full_policy"},
        "dense_relevance": {
            "available": bool(caps.get("dense_relevance")),
            "scope": "observed_relevance_not_randomized_policy"},
        "controlled_parameter": {
            "available": bool(caps.get("replayable_recommender")),
            "scope": "owned_recommender_bounded_parameters_only"},
        "sequential_intervention": {
            "available": bool(caps.get("sequential_intervention")),
            "scope": "requires_explicit_stateful_replay_protocol"},
        "policy_comparison": {
            "available": bool(caps.get("policy_comparison")),
            "scope": "requires_two_executable_fixed-candidate_policies"},
    }


def assess_candidates(tools, concepts: list[dict],
                      candidates: list[dict]) -> list[dict]:
    """Check support without observing outcomes or adapting to observed effects.

    Explanations and experiments originate with the agent. Neither this scorer
    nor an adapter introduces a target Bias name or intervention parameter.
    """
    active = {c["id"] for c in concepts
              if c["status"] not in ("design_blocked", "rejected_development")}
    capabilities = available_capabilities(tools.adapter)
    checked = []
    seen = set()
    for candidate in candidates:
        kind = candidate["required_capability"]
        record = {"candidate": candidate, "score": 0, "status": "",
                  "reason": "", "evidence_scope": None,
                  "scientific_scope": "exploratory_unidentified"}
        if candidate["concept_id"] not in active:
            record.update(status="invalid_concept", reason="Unknown or inactive concept")
        elif (candidate["direction_h1"] == candidate["direction_h2"] or
              not candidate["prediction_h1"].strip() or
              not candidate["prediction_h2"].strip() or
              not candidate["target_quantity"].strip()):
            record.update(status="not_discriminating",
                          reason="H1 and H2 lack distinct predictions for one target")
        elif kind not in capabilities or not capabilities[kind]["available"]:
            record.update(status="not_executable_with_current_capabilities",
                          reason="This dataset/recommender cannot execute " + kind)
        elif kind != "sql_observation":
            # Scientific capabilities exist, but a prose candidate is NOT a
            # full runnable intervention/estimand specification.
            record.update(status="requires_executable_protocol",
                          reason="Use the specialized Workbench/reference executor "
                                 "with an explicit registered estimand")
        elif not candidate["sql"].strip():
            record.update(status="invalid_sql", reason="Missing SQL observation")
        elif previously_measured(candidate["sql"], tools.history):
            record.update(status="already_measured",
                          reason="Existing observation can be reused")
        elif candidate["sql"].strip() in seen:
            record.update(status="duplicate_candidate", reason="Same proposed SQL")
        else:
            seen.add(candidate["sql"].strip())
            overlap = (audit_construct_overlap(
                tools, candidate["predictor_column"], candidate["outcome_column"])
                if candidate.get("predictor_column") and candidate.get("outcome_column")
                else audit_observation_sql(tools, candidate["sql"]))
            if overlap and overlap["status"] == "overlapping_constructs":
                record.update(
                    status="overlapping_constructs",
                    reason="Predictor and outcome reuse the same measured history "
                           "events; redesign outcome or separate temporal windows",
                    lineage_audit=overlap)
            else:
                try:
                    tools.preflight_sql(candidate["sql"])
                except Exception as exc:
                    record.update(status="invalid_sql", reason=str(exc)[:240])
                else:
                    disagreement = abs(
                        DIRECTIONS[candidate["direction_h1"]] -
                        DIRECTIONS[candidate["direction_h2"]])
                    decision = candidate.get("decision_target", "unknown")
                    reference = candidate.get("reference_kind", "none")
                    # A model's claimed "ranking/allocation/reference" is
                    # NOT itself evidence of what the SQL measures. Reward
                    # only the restricted matched-pool exposure diagnostic
                    # for which we can inspect actual source/aggregate fields.
                    query = candidate["sql"].lower()
                    eligible_reference = (
                        reference == "random_insertions"
                        and bool(tools.adapter.capabilities.get(
                            "randomized_exposure_probe"))
                        and "shared_item_pool_events" in getattr(
                            tools, "reference_views", [])
                        and "shared_item_pool_events" in query)
                    observes_allocation = (
                        decision == "exposure_allocation"
                        and eligible_reference
                        and "is_random" in query
                        and any(field in query for field in
                                getattr(tools.adapter, "item_features", ()))
                        and "count(" in query)
                    # This controls ordering only. Even the recognized
                    # diagnostic is NOT a normative or causal Bias reference.
                    priority = 3 if observes_allocation else 0
                    record.update(
                        status="feasible_descriptive_observation",
                        evidence_scope=capabilities[kind]["scope"],
                        decision_reference_priority=priority,
                        decision_link_audit={
                            "declared": decision,
                            "measured_restricted_allocation": observes_allocation,
                            "reference_scope_grounded": eligible_reference,
                        },
                        score=disagreement,
                        scientific_scope=(
                            "restricted_exposure_reference_diagnostic_only"
                            if observes_allocation else
                            "exploratory_unidentified"))
        checked.append(record)
    return checked


def choose_feasible(checked: list[dict]) -> dict | None:
    """Maximum *declared* H1/H2 disagreement, not largest observed effect."""
    ranked = [(index, result) for index, result in enumerate(checked)
              if result["status"] == "feasible_descriptive_observation"]
    return (max(ranked, key=lambda indexed: (
        indexed[1].get("decision_reference_priority", 0),
        indexed[1]["score"], -indexed[0]))[1] if ranked else None)


def design_and_select(planner, context: dict, tools, concepts: list[dict], tag: str):
    """At most one pre-outcome repair; unsupported causal capabilities are final."""
    initial = planner.act(context, tag, CANDIDATE_SCHEMA)
    first = assess_candidates(tools, concepts, initial["candidates"])
    attempts = [first]
    chosen = choose_feasible(first)
    if chosen is None and any(
            x["status"] in ("invalid_sql", "not_discriminating",
                             "overlapping_constructs") for x in first):
        repaired = planner.act({
            **context, "rejected_design_audit": first,
            "revision_requirement": (
                "Fix invalid SQL or overlapping history constructs using a "
                "different outcome/time window. Keep H1/H2 tied to the same "
                "measurable decision, and do not merely flip a direction label. "
                "A behavior correlation does not measure ranking decisions. "
                "Unsupported interventions remain unsupported."),
        }, tag + "-repair", CANDIDATE_SCHEMA)
        checked = assess_candidates(tools, concepts, repaired["candidates"])
        attempts.append(checked)
        chosen = choose_feasible(checked)
    return chosen, attempts[-1], attempts


def execute_observation(tools, picked: dict) -> dict:
    """Execute only the selected bounded SQL; never issue a Bias verdict."""
    if picked["status"] != "feasible_descriptive_observation":
        raise ValueError("Cannot execute an unsupported experiment")
    evidence = tools.sql(picked["candidate"]["sql"])
    return {
        "experiment_kind": "sql_observation",
        "concept_id": picked["candidate"]["concept_id"],
        "target_quantity": picked["candidate"]["target_quantity"],
        "h1_prediction": picked["candidate"]["prediction_h1"],
        "h2_prediction": picked["candidate"]["prediction_h2"],
        "predicted_directions": {
            "h1": picked["candidate"]["direction_h1"],
            "h2": picked["candidate"]["direction_h2"]},
        "executed_evidence": evidence,
        "scientific_status": "exploratory_descriptive_not_h1h2_confirmation",
        "h1_h2_discriminated": False,
        "reference_identified": False,
        "production_algorithm_causality_identified": False,
        "design_scientific_scope": picked["scientific_scope"],
        "next_requirement": "Pre-register a supported inferential test or "
                            "controlled reference before adjudicating mechanism",
    }
