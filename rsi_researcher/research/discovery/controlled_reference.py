"""Agent-specified interventions on an owned recommender and its frozen panel.

The parameter is chosen by the research agent, not by a Bias taxonomy. The
intervention identifies a difference in THIS executable model's offline metric,
not in the unknown KuaiRec production policy. Confirmation users stay sealed.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from random import Random
from statistics import mean

from rsi_researcher.research.discovery.real_intervention import (
    K, WINDOW, PROFILE_STRENGTH, read_data, rank,
)

PARAMETER_BOUNDS = {
    "history_window": (5, 120),
    "recent_write_drop": (0, 40),
    "profile_strength": (0.0, 2.0),
    "prior_weight": (0.0, 2.0),
    "positive_feedback_weight": (0.0, 3.0),
    "negative_feedback_weight": (0.0, 3.0),
}
INTEGER_PARAMETERS = {"history_window", "recent_write_drop"}
OUTCOMES = ("ndcg@10", "recall@10")

CONTROLLED_EXPERIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "minLength": 12},
        "parameter": {"type": "string", "enum": list(PARAMETER_BOUNDS)},
        "control_value": {"type": "number"},
        "treatment_value": {"type": "number"},
        "outcome": {"type": "string", "enum": list(OUTCOMES)},
        "prediction_if_mechanism": {"type": "string", "enum": [
            "increase", "decrease", "no_change"]},
        "prediction_if_rival": {"type": "string", "enum": [
            "increase", "decrease", "no_change"]},
        "meaningful_margin": {"type": "number", "minimum": 0, "maximum": 1},
        "reference_justification": {"type": "string", "minLength": 12},
        # These are required when the model experiment is linked to a
        # real development-log observation through Open Discovery.
        "observed_group_feature": {"type": "string", "minLength": 1},
        "mechanism_link": {"type": "string", "minLength": 20},
    },
    "required": ["question", "parameter", "control_value", "treatment_value",
                 "outcome", "prediction_if_mechanism", "prediction_if_rival",
                 "meaningful_margin", "reference_justification"],
    "additionalProperties": False,
}


def validate_controlled_spec(spec):
    from jsonschema import Draft202012Validator
    Draft202012Validator(CONTROLLED_EXPERIMENT_SCHEMA).validate(spec)
    name = spec["parameter"]
    low, high = PARAMETER_BOUNDS[name]
    for key in ("control_value", "treatment_value"):
        value = spec[key]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be finite and in {low}..{high} for {name}")
        if name in INTEGER_PARAMETERS and not float(value).is_integer():
            raise ValueError(f"{name} requires an integer value")
    if spec["control_value"] == spec["treatment_value"]:
        raise ValueError("Controlled experiment must change one mechanism parameter")
    if spec["prediction_if_mechanism"] == spec["prediction_if_rival"]:
        raise ValueError("H1/H2 must predict different outcomes on the same intervention")


def _policy(history, spec, value):
    parameters = {"history_window": WINDOW,
                  "recent_write_drop": 0,
                  "profile_strength": PROFILE_STRENGTH,
                  "prior_weight": 1.0,
                  "positive_feedback_weight": 1.0,
                  "negative_feedback_weight": 1.0}
    parameters[spec["parameter"]] = value
    tail_drop = int(parameters.pop("recent_write_drop"))
    if tail_drop:
        history = history[:-tail_drop]
    parameters["history_window"] = int(parameters["history_window"])
    return history, parameters


def _utility(top, positives, metric):
    if not positives:
        return None
    if metric == "recall@10":
        return sum(label for _, _, label in top) / positives
    actual = sum(label / math.log2(position + 2)
                 for position, (_, _, label) in enumerate(top))
    ideal = sum(1.0 / math.log2(position + 2)
                for position in range(min(K, positives)))
    return actual / ideal


def evaluate_controlled(zip_path: str, spec: dict, *,
                        phase: str = "development") -> dict:
    """One prespecified paired contrast on development OR sealed confirmation users.

    The panel is seen by the evaluator for scoring, never by the policy
    and never copied to the model-facing development database.
    """
    validate_controlled_spec(spec)
    categories, priors, histories, panels, folds = read_data(zip_path, phase)
    rows = []
    tested_placebo_users = 0
    for uid, candidates in sorted(panels.items()):
        history = sorted(histories[uid])
        if len(candidates) < K or len(history) < WINDOW + 5:
            continue
        values = []
        ranking_ids = []
        placebo_equal = True
        has_old_writes = False
        for value in (spec["control_value"], spec["treatment_value"]):
            state_events, params = _policy(history, spec, value)
            ranked, positive = rank(candidates, state_events, categories, priors, **params)
            metric = _utility(ranked, positive, spec["outcome"])
            if metric is None or len(ranked) != K:
                break
            values.append(metric)
            ranking_ids.append(tuple(vid for _, vid, _ in ranked))
            # An old write outside the effective state window must not matter.
            # This is a pipeline negative control, not a rival-mechanism test.
            keep = params["history_window"]
            shorter = state_events[-keep:]
            placebo, _ = rank(candidates, shorter, categories, priors, **params)
            placebo_equal &= (tuple(vid for _, vid, _ in placebo) ==
                              ranking_ids[-1])
            has_old_writes |= len(state_events) > keep
        if len(values) != 2:
            continue
        if not placebo_equal:
            raise ValueError("Negative-control history outside the model window changed ranking")
        tested_placebo_users += int(has_old_writes)
        rows.append({"uid": uid, "control": values[0],
                     "treatment": values[1],
                     "difference": values[1] - values[0],
                     "ranking_changed": ranking_ids[0] != ranking_ids[1]})
    if len(rows) < 12:
        raise ValueError(f"Insufficient independent {phase} users: {len(rows)}")
    differences = [x["difference"] for x in rows]
    rng = Random(20261028)
    boot = sorted(mean(rng.choices(differences, k=len(differences)))
                  for _ in range(400))
    result = {
        "estimate": mean(differences),
        "ci95_bootstrap": [boot[10], boot[389]],
        "n_paired_units": len(rows),
        "n_independent_units": len(rows),
        "mean_control": mean(x["control"] for x in rows),
        "mean_treatment": mean(x["treatment"] for x in rows),
        "ranking_change_fraction": mean(x["ranking_changed"] for x in rows),
        "outcome": spec["outcome"],
        "negative_control_tested": tested_placebo_users > 0,
        "negative_control_passed": True if tested_placebo_users > 0 else None,
        "negative_control_users_with_old_writes": tested_placebo_users,
    }
    estimand = {
        "population": phase + "_users_with_sufficient_history_and_fixed_label_panel",
        "statistical_unit": "uid",
        "treatment": {
            "parameter": spec["parameter"],
            "control": spec["control_value"],
            "intervention": spec["treatment_value"],
        },
        "outcome": spec["outcome"],
        "quantity": "mean_per_user_treatment_minus_control",
        "candidate_reference": "same_" + phase + "_user_masked_item_evaluation_panel",
        "relevance_reference": "observed_watch_ratio_ge_one",
        "phase": phase,
    }
    return {
        "specification": dict(spec),
        "estimand": estimand,
        "result": result,
        "evidence_scope": "constructed_recommender_" + phase + "_only",
        "reference_source": "KuaiRec near-full observed watch_ratio>=1 evaluation panel",
        "identification": {
            "within_model_paired_intervention": True,
            "same_users": True,
            "same_candidates": True,
            "same_heldout_relevance_labels": True,
            "train_users_separate_from_dev": True,
            "confirmation_users_consulted": phase == "confirmation",
            "platform_policy_identified": False,
            "novel_real_world_bias_identified": False,
            "reference_status": "identified_model_effect_only",
        },
        "model_parameters": {k: [spec["control_value"], spec["treatment_value"]]
                             if k == spec["parameter"] else None
                             for k in PARAMETER_BOUNDS},
        "user_folds": dict(folds),
        # Per-user details stay in the machine journal, never model prompt.
        "per_user": rows,
    }

def confirm_frozen_controlled(zip_path: str, frozen: dict) -> dict:
    """Independently replay a PRE-FROZEN model mechanism on sealed users.

    No agent, SQL discovery session or development statistics are consulted
    while selecting this experiment. A positive result is explicitly limited
    to the owned recommender; it cannot certify an unseen production policy.
    """
    if frozen.get("status") != "frozen_scoped_mechanism_unconfirmed":
        raise ValueError("Only a frozen scoped model mechanism can be confirmed")
    identifier = frozen.get("frozen_id")
    concept = next((c for c in frozen.get("concepts", [])
                    if c.get("id") == identifier), None)
    opportunity = next((r for r in frozen.get(
        "research_opportunities", {}).get("opportunities", [])
                        if r.get("id") == identifier), None)
    if concept is None or opportunity is None:
        raise ValueError("Frozen concept and reference provenance must be present")
    from rsi_researcher.research.discovery.references import (
        scoped_model_freeze_gate, eligible_scoped_model_trials, rival_prediction_audit)
    # Recheck the evidence gate from saved development artifacts, instead of
    # trusting a mutable status string or an agent's narrative.
    gate = scoped_model_freeze_gate(concept, opportunity, enough_exploration=True)
    if not all(gate.values()):
        raise ValueError("Frozen model mechanism lacks required development evidence")
    # The sealed replay must use the SAME fully qualified trial that passed
    # development review, not merely the last trial labeled H1.
    trials = eligible_scoped_model_trials(concept, opportunity)
    frozen_trial = trials[-1]
    specification = frozen_trial["specification"]
    validate_controlled_spec(specification)
    confirmation = evaluate_controlled(zip_path, specification, phase="confirmation")
    audit = rival_prediction_audit(
        confirmation["result"],
        specification["prediction_if_mechanism"],
        specification["prediction_if_rival"],
        specification["meaningful_margin"])
    ci = confirmation["result"]["ci95_bootstrap"]
    supported = bool(audit["distinguishes_predictions"]
                     and audit["supports"] == "H1"
                     and (ci[0] > 0 or ci[1] < 0)
                     and confirmation["result"]["negative_control_tested"]
                     and confirmation["result"]["negative_control_passed"] is True)
    return {
        "status": ("confirmed_scoped_model_mechanism" if supported
                   else "scoped_model_mechanism_not_confirmed"),
        "frozen_id": identifier,
        "tested_concept_version": concept["version"],
        "frozen_specification": specification,
        "frozen_observational_estimand": frozen_trial.get(
            "observational_anchor", {}).get("observed_estimand"),
        "frozen_source_evidence_scope": frozen_trial.get(
            "observational_anchor", {}).get("source_evidence_scope"),
        "confirmation_measurement": {
            "estimand": confirmation["estimand"],
            "result": confirmation["result"],
            "identification": confirmation["identification"],
            "evidence_scope": confirmation["evidence_scope"],
        },
        "confirmation_prediction_audit": audit,
        "independent_confirmation": supported,
        "confirmation_scope": "constructed_recommender_only",
        "platform_causal_bias_established": False,
        "real_world_novelty_established": False,
    }
