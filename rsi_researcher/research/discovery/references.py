"""Data-agnostic reference and estimand contracts.

Scientific declarations are hypotheses. Only executed evidence may change status.
This module validates provenance and alignment; it does not select Bias classes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, isclose


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


def constrain_controlled_critique(critique: dict, model_audit: dict,
                                  source_anchor: dict) -> dict:
    """A language-only 'ready' must not overrule executed rival predictions.

    A numerical contradiction can still distinguish rivals scientifically,
    but it does not justify freezing H1. Preserve the Critic's other fields;
    the deterministic override applies only to a premature ready verdict.
    """
    model_support = model_audit.get("supports")
    source_supported = source_anchor.get("observed_h1_supported") is True
    if critique.get("decision") != "ready" or (
            model_support == "H1" and source_supported):
        return critique
    decision = "revise" if model_support == "H2" else "test"
    return {
        **critique,
        "decision": decision,
        "reason": (critique.get("reason", "") +
                   " [NUMERICAL VETO: model supports " +
                   str(model_support) + "; observed source supports H1=" +
                   str(source_supported) + ". Freeze cannot proceed.]"),
        "numeric_ready_veto": True,
    }


def scoped_trial_evidence_audit(trial: dict, opportunity: dict) -> dict:
    """Recompute scientific support from ONE trial, not trusted LLM status flags.

    The observed-log estimand and the model intervention are DIFFERENT
    quantities. A linked source signal is a prerequisite to testing a
    constructed-model analog, never proof of causal transfer to a platform.
    """
    spec = trial.get("specification", {})
    stats = trial.get("statistics", {})
    ident = trial.get("identification", {})
    anchor = trial.get("observational_anchor", {})
    selection = opportunity.get("selection_contract", {})
    chosen_estimand = selection.get("estimand_spec", {})
    observed_estimand = anchor.get("observed_estimand", {})
    model_ci = stats.get("ci95_bootstrap") or []
    model_numbers = [stats.get(k) for k in
                     ("estimate", "mean_control", "mean_treatment")]
    model_n = stats.get("n_paired_units", 0)
    valid_model_numbers = bool(
        len(model_ci) == 2
        and all(isinstance(x, (int, float)) and isfinite(x)
                for x in [*model_numbers, *model_ci])
        and model_ci[0] <= model_ci[1]
        and isinstance(model_n, int) and model_n >= 12
        and isclose(model_numbers[2] - model_numbers[1],
                    model_numbers[0], abs_tol=1e-9))
    try:
        prediction = rival_prediction_audit(
            stats, spec["prediction_if_mechanism"],
            spec["prediction_if_rival"], spec["meaningful_margin"])
    except (KeyError, TypeError, ValueError):
        prediction = {"supports": "inconclusive", "distinguishes_predictions": False}
    stored_prediction = trial.get("prediction_audit", {})
    model_supported = bool(
        valid_model_numbers and prediction.get("supports") == "H1"
        and prediction.get("distinguishes_predictions")
        and prediction.get("observed_direction") in ("increase", "decrease")
        and all(stored_prediction.get(k) == prediction.get(k)
                for k in ("supports", "observed_direction",
                          "distinguishes_predictions", "prediction_h1",
                          "prediction_h2")))

    source_stats = anchor.get("observed_statistics") or {}
    source_ci = source_stats.get("ci95_bootstrap") or []
    source_numbers = [source_stats.get(k) for k in
                      ("estimate", "mean_control", "mean_treatment")]
    valid_source_numbers = bool(
        len(source_ci) == 2
        and all(isinstance(x, (int, float)) and isfinite(x)
                for x in [*source_numbers, *source_ci])
        and source_ci[0] <= source_ci[1])
    # Paired-user mean differences need not equal pooled-group mean differences.
    # Workbench's per-user estimator and pooled means weight users differently.
    try:
        observed_prediction = rival_prediction_audit(
            source_stats, selection["prediction_if_mechanism"],
            selection["prediction_if_rival"], 0.01)
    except (KeyError, TypeError, ValueError):
        observed_prediction = {"supports": "inconclusive",
                               "distinguishes_predictions": False}
    stored_source = anchor.get("observed_prediction_audit", {})
    source_supported = bool(
        valid_source_numbers and observed_prediction.get("supports") == "H1"
        and observed_prediction.get("observed_direction") in ("increase", "decrease")
        and observed_prediction.get("distinguishes_predictions")
        and all(stored_source.get(k) == observed_prediction.get(k)
                for k in ("supports", "observed_direction",
                          "distinguishes_predictions", "prediction_h1",
                          "prediction_h2"))
        and anchor.get("observed_h1_supported") is True)

    same_source = bool(
        chosen_estimand and observed_estimand == chosen_estimand
        and anchor.get("same_selected_estimand")
        and anchor.get("source_observed")
        and anchor.get("source_evidence_scope") == "development_observed_responses_only"
        and anchor.get("observed_executed_sql"))
    link = bool(
        same_source
        and anchor.get("mechanism_link_declared")
        and spec.get("observed_group_feature") == chosen_estimand.get("group_expression")
        and anchor.get("agent_selected_model_parameter") == spec.get("parameter")
        and isinstance(anchor.get("agent_proposed_mechanism_link"), str)
        and len(anchor["agent_proposed_mechanism_link"].strip()) >= 20
        and not anchor.get("cross_environment_causal_transfer_identified")
        and not anchor.get("user_preference_rival_ruled_out"))
    model_scope = bool(
        ident.get("within_model_paired_intervention")
        and ident.get("same_users") and ident.get("same_candidates")
        and ident.get("same_heldout_relevance_labels")
        and ident.get("train_users_separate_from_dev")
        and not ident.get("confirmation_users_consulted")
        and not ident.get("platform_policy_identified")
        and trial.get("estimand", {}).get("phase") == "development"
        and trial.get("estimand", {}).get("outcome") == spec.get("outcome")
        and stats.get("outcome") == spec.get("outcome")
        and trial.get("estimand", {}).get("treatment") == {
            "parameter": spec.get("parameter"),
            "control": spec.get("control_value"),
            "intervention": spec.get("treatment_value")})
    return {
        "executed_h1_rival_discrimination": model_supported and model_scope,
        "agent_mechanism_linked_to_observed_estimand": source_supported and link,
        # Only a state-window *pipeline invariant*, NOT a rival-mechanism test.
        "pipeline_invariant_checked": bool(
            stats.get("negative_control_tested")
            and stats.get("negative_control_passed") is True),
    }


def latest_scoped_trial_index(concept: dict, opportunity: dict) -> int | None:
    """Use the latest trial for this hypothesis version; never cherry-pick."""
    indexed = [
        i for i, trial in enumerate(opportunity.get("controlled_reference_tests", []))
        if trial.get("tested_concept_version") == concept.get("version")]
    if not indexed:
        return None
    trial = opportunity["controlled_reference_tests"][indexed[-1]]
    return indexed[-1] if all(scoped_trial_evidence_audit(trial, opportunity).values()) else None


def scoped_model_freeze_gate(concept: dict, opportunity: dict, *,
                             enough_exploration: bool) -> dict:
    """Freeze a single independently auditable owned-model trial, not platform Bias."""
    bias = opportunity.get("bias_relevance", {})
    valid_trial = latest_scoped_trial_index(concept, opportunity)
    critique = concept.get("critique") or {}
    return {
        "reference_is_owned_model": bias.get("reference_spec", {}).get("kind") == "controlled_model",
        "current_concept_version_tested": any(
            t.get("tested_concept_version") == concept.get("version")
            for t in opportunity.get("controlled_reference_tests", [])),
        "executed_h1_rival_discrimination": valid_trial is not None,
        "agent_mechanism_linked_to_observed_estimand": valid_trial is not None,
        "exploration_completed": enough_exploration,
        "critic_ready_for_scoped_mechanism": critique.get("decision") == "ready",
        "critic_observable_valid": bool(critique.get("observable_valid")),
        "critic_rival_discriminated": bool(critique.get("rival_discriminated")),
        "literature_checked": concept.get("literature") is not None,
    }
