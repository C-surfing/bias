"""Data-agnostic reference and estimand contracts.

Scientific declarations are hypotheses. Only executed evidence may change status.
This module validates provenance and alignment; it does not select Bias classes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class ReferenceSpec:
    kind: str
    target_quantity: str
    target_reference: str
    affected_population: str
    identification_assumptions: str
    source_scope: str
    status: str = "proposed_unidentified"

    @classmethod
    def from_choice(cls, choice: dict, capabilities: dict) -> "ReferenceSpec":
        kind = choice.get("reference_kind", "none")
        capability = {
            "randomized_exposure": "randomized_exposure_probe",
            "dense_relevance": "dense_relevance",
            "controlled_model": "replayable_recommender",
        }.get(kind)
        if kind not in ("none", "randomized_exposure", "dense_relevance", "controlled_model"):
            raise ValueError("Unsupported reference type")
        if capability and not capabilities.get(capability, False):
            raise ValueError(f"Dataset does not support {kind} reference; "
                             "measure another reference or document the gap")
        return cls(
            kind=kind,
            target_quantity=choice.get("proposed_distortion", ""),
            target_reference=choice.get("target_reference", ""),
            affected_population=choice.get("contrast_filter", ""),
            identification_assumptions=choice.get("reference_identification", ""),
            source_scope={
                "randomized_exposure": "restricted_random_insertion_not_normative",
                "dense_relevance": "observed_relevance_not_platform_allocation",
                "controlled_model": "constructed_recommender_only",
                "none": "unidentified",
            }[kind])

    def as_record(self):
        return asdict(self)


@dataclass(frozen=True)
class EstimandSpec:
    source_table: str
    unit_expression: str
    population_filter: str
    group_expression: str
    control_value: str
    treatment_value: str
    outcome_expression: str
    analysis: str

    @classmethod
    def from_choice(cls, choice: dict) -> "EstimandSpec":
        return cls(
            source_table=choice["contrast_source"],
            unit_expression="uid",
            population_filter=choice.get("contrast_filter", ""),
            group_expression=choice["contrast_group"],
            control_value=choice["control_value"],
            treatment_value=choice["treatment_value"],
            outcome_expression=choice["contrast_outcome"],
            analysis=choice["analysis"])

    def compare_executed(self, executed: dict) -> dict:
        """Distinguish exact target alignment from a conditional sensitivity test."""
        declared = asdict(self)
        differences = {k: [v, executed.get(k)] for k, v in declared.items()
                       if str(v).strip() != str(executed.get(k, "")).strip()}
        return {
            "same_estimand": not differences,
            "same_core_contrast": all(k == "population_filter" for k in differences),
            "population_shifted": "population_filter" in differences,
            "differences": differences,
        }

    def as_record(self):
        return asdict(self)


def observed_direction(statistics: dict, meaningful_margin: float) -> str:
    """Numerical evidence only; an insignificant effect is NOT 'no_change'."""
    ci = statistics.get("ci95_bootstrap") or ()
    if len(ci) != 2 or meaningful_margin < 0:
        return "inconclusive"
    low, high = map(float, ci)
    if not all(isfinite(v) for v in (low, high)) or low > high:
        return "inconclusive"
    if low > meaningful_margin:
        return "increase"
    if high < -meaningful_margin:
        return "decrease"
    if meaningful_margin > 0 and -meaningful_margin < low <= high < meaningful_margin:
        return "no_change"
    return "inconclusive"


def rival_prediction_audit(statistics: dict, h1: str, h2: str,
                           margin: float) -> dict:
    """Evaluate directions only for a valid executed same-target experiment."""
    direction = observed_direction(statistics, margin)
    return {
        "observed_direction": direction,
        "prediction_h1": h1,
        "prediction_h2": h2,
        "distinguishes_predictions": bool(
            direction != "inconclusive" and h1 != h2
            and direction in (h1, h2)),
        "supports": ("H1" if direction == h1 and h1 != h2 else
                     "H2" if direction == h2 and h1 != h2 else "inconclusive"),
        "note": "Observational prediction agreement does not identify a causal Bias.",
    }


def audit_reference_probe(choice: dict, probe: dict) -> dict:
    """Bind an executed randomized-mode diagnostic to the chosen observable.

    The target is the *response-gap interaction* between standard and random
    insertion within a shared item pool, not the observational arm gap. Even a
    decisive prediction does NOT identify an optimal allocation policy.
    """
    expected = {
        "source_table": choice["contrast_source"],
        "group_expression": choice["contrast_group"],
        "outcome_expression": choice["contrast_outcome"],
        "control_value": choice["control_value"],
        "treatment_value": choice["treatment_value"],
        "population_filter": choice.get("contrast_filter", "") or "TRUE",
    }
    differences = {key: (value, probe.get(key)) for key, value in expected.items()
                   if str(value).strip() != str(probe.get(key, "")).strip()}
    ci = probe.get("policy_response_interaction_ci95_user_bootstrap", [])
    measured = {
        "estimate": probe.get("standard_minus_random_response_contrast"),
        "ci95_bootstrap": ci,
    }
    predictions = rival_prediction_audit(
        measured, choice["mechanism_direction"], choice["rival_direction"],
        choice.get("meaningful_margin", 0.01))
    pool_valid = (probe.get("matched_item_day_support") is True
                  and probe.get("matched_item_day_strata", 0) > 0
                  and probe.get("reference_pool_scope") ==
                      "shared_item_pool_same_video_same_day_same_ui_scope")
    aligned = not differences and pool_valid
    return {
        "estimand": "standard_minus_random_item_group_response_interaction",
        "scope": "observed_standard_vs_random_insertion_modes",
        "matches_selected_observable": aligned,
        "matched_item_day_reference_valid": pool_valid,
        "differences": differences,
        "prediction_audit": predictions if aligned else {
            **predictions, "supports": "inconclusive", "distinguishes_predictions": False},
        "reference_identifies_production_bias": False,
        "status": ("measured_restricted_reference_only" if aligned
                   else "estimand_mismatch"),
        "reason": ("Mode-conditioned response comparison does not randomize the "
                   "production recommender or user preferences."),
    }


def eligible_scoped_model_trials(concept: dict, opportunity: dict) -> list[dict]:
    """One trial must satisfy BOTH the controlled result and the source bridge.

    In particular, an H1 effect from trial A must never be joined to an
    observational bridge from trial B. The returned order is execution order,
    so freezing and sealed confirmation select the same last qualifying trial.
    """
    eligible = []
    for trial in opportunity.get("controlled_reference_tests", []):
        if trial.get("tested_concept_version") != concept.get("version"):
            continue
        stats = trial.get("statistics", {})
        audit = trial.get("prediction_audit", {})
        identification = trial.get("identification", {})
        anchor = trial.get("observational_anchor", {})
        ci = stats.get("ci95_bootstrap") or []
        model_valid = (
            audit.get("distinguishes_predictions") is True
            and audit.get("supports") == "H1"
            and len(ci) == 2 and (ci[0] > 0 or ci[1] < 0)
            and stats.get("negative_control_tested") is True
            and stats.get("negative_control_passed") is True
            and identification.get("within_model_paired_intervention") is True
            and not identification.get("platform_policy_identified")
            and not identification.get("confirmation_users_consulted"))
        bridge_valid = (
            anchor.get("source_observed") is True
            and anchor.get("same_selected_estimand") is True
            and anchor.get("mechanism_link_declared") is True
            and anchor.get("observed_h1_supported") is True
            and anchor.get("cross_environment_causal_transfer_identified") is False)
        if model_valid and bridge_valid and trial.get("specification"):
            eligible.append(trial)
    return eligible


def scoped_model_freeze_gate(concept: dict, opportunity: dict, *,
                             enough_exploration: bool) -> dict:
    """Freeze an evidence-linked *owned-model* mechanism, never platform Bias."""
    bias = opportunity.get("bias_relevance", {})
    tests = [t for t in opportunity.get("controlled_reference_tests", [])
             if t.get("tested_concept_version") == concept.get("version")]
    complete = eligible_scoped_model_trials(concept, opportunity)
    critique = concept.get("critique") or {}
    return {
        "reference_is_owned_model": bias.get("reference_spec", {}).get("kind") == "controlled_model",
        "current_concept_version_tested": bool(tests),
        "executed_h1_rival_discrimination": bool(complete),
        "agent_mechanism_linked_to_observed_estimand": bool(complete),
        "same_trial_has_complete_evidence": bool(complete),
        "exploration_completed": enough_exploration,
        "critic_ready_for_scoped_mechanism": critique.get("decision") == "ready",
        "critic_observable_valid": bool(critique.get("observable_valid")),
        "critic_rival_discriminated": bool(critique.get("rival_discriminated")),
        "literature_checked": concept.get("literature") is not None,
    }
