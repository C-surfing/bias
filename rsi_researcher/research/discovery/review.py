"""Evidence-based Discovery review: workflow validity != scientific validity.

The review never reads an independent holdout or grades a free-text mechanism
as true. The private evaluator must separately test hidden predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess(journal: list[dict], state: dict) -> dict:
    derived = [event["derived_feature"] for event in journal
               if "derived_feature" in event]
    executed = [event for event in journal if "experiment" in event]
    referenced = [event for event in executed if event.get(
        "feature_lineage", {}).get("used")]
    # Merely changing prose or adding strata without changing the estimand's
    # source, population, group or outcome is a supplementary analysis,
    # not a genuinely different rival-discriminating numeric experiment.
    material_keys = ("source_query", "population_filter", "group_expression",
                     "control_value", "treatment_value", "outcome_expression")
    def materially_different_numeric(a: dict, b: dict) -> bool:
        # Changing an unused filter, query wording, or adding a cosmetic
        # stratum is not independent numeric falsification evidence.
        quantities = ("estimate", "mean_control", "mean_treatment",
                      "n_control", "n_treatment", "n_paired_units")
        if any(a.get(k) != b.get(k) for k in quantities):
            return True
        return a.get("ci95_bootstrap") != b.get("ci95_bootstrap")

    critic_workbench = [
        event for event in journal if "critic_experiment" in event
        and "experiment" in event
        and any(event["experiment"]["specification"].get(k)
                != event["critic_experiment"]["specification"].get(k)
                for k in material_keys)
        and materially_different_numeric(event["experiment"]["result"],
                                         event["critic_experiment"]["result"])]
    critic_sql = [event for event in journal
                  if "critic_sql_countertest" in event]
    revised = [event for event in journal if "revised_concept" in event
               and "reflection" in event]
    abandoned = [event for event in journal
                 if event.get("reflection", {}).get("decision") == "abandon"]
    explicit_opportunity_selections = [e for e in journal if "opportunity_selection" in e]
    opportunity_bound_tests = [e for e in executed if e.get("feature_lineage", {}).get("used")]
    checked_selections = [e for e in explicit_opportunity_selections
                          if e.get("selection_feasibility", {}).get("valid_support")
                          and e["opportunity_selection"].get("mechanism_direction") !=
                              e["opportunity_selection"].get("rival_direction")
                          and e.get("selection_feasibility", {}).get("outcome_numeric")]
    aligned = [e for e in executed if e.get("selection_contract_audit", {}).get(
        "matches_original_contrast")]
    checks = {
        "agent_explicitly_selected_opportunity": bool(explicit_opportunity_selections),
        "rival_predictions_disagree_and_support_is_measured": bool(checked_selections),
        "selected_contrast_reaches_workbench": bool(aligned),
        "opportunity_selection_has_numeric_experiment": bool(opportunity_bound_tests),
        "agent_computed_derived_feature": bool(derived),
        "experiment_references_registered_feature": bool(referenced),
        "workbench_executed_user_level_test": bool(executed),
        "critic_executed_distinct_workbench_test": bool(critic_workbench),
        "critic_executed_descriptive_sql_probe": bool(critic_sql),
        "agent_substantively_revised_hypothesis": bool(revised),
        "agent_abandoned_hypothesis_after_countertest": bool(abandoned),
    }
    accepted = all(checks[key] for key in (
        "agent_computed_derived_feature",
        "experiment_references_registered_feature",
        "workbench_executed_user_level_test",
        "critic_executed_distinct_workbench_test")) and (
        checks["agent_substantively_revised_hypothesis"]
        or checks["agent_abandoned_hypothesis_after_countertest"])
    # Backward-compatible audit for runs before opportunity selection existed.
    if explicit_opportunity_selections:
        accepted = (accepted and
                    checks["rival_predictions_disagree_and_support_is_measured"] and
                    checks["selected_contrast_reaches_workbench"])
    return {
        "full_qwen_workflow_accepted": accepted,
        "checks": checks,
        "counts": {"explicit_opportunity_selections": len(explicit_opportunity_selections),
                   "opportunity_bound_experiments": len(opportunity_bound_tests),
                   "derived_features": len(derived),
                   "successful_workbench_experiments": len(executed),
                   "feature_bound_workbench_experiments": len(referenced),
                   "critic_workbench_tests": len(critic_workbench),
                   "critic_sql_only_tests": len(critic_sql),
                   "revisions": len(revised),
                   "abandonments": len(abandoned)},
        "concepts": [
            {"id": c["id"], "version": c["version"], "status": c["status"],
             "mechanism": c["claim"]["mechanism"],
             "rival": c["claim"]["rival"],
             "falsifier": c["claim"]["falsifier"],
             "measurement": c["claim"]["measurement"]}
            for c in state.get("concepts", [])],
        "scientific_quality": {
            "observed_associations_are_not_causal_bias": True,
            "randomized_reference_diagnostic_executed": any(
                "randomized_reference_diagnostic" in e for e in journal),
            "bias_reference_discrepancy_verified": False,
            "distortion_mechanism_identified": False,
            "independently_confirmed_recommender_bias": False,
            "independent_mechanism_confirmation": "not_evaluated",
            "hidden_condition_consequence_agreement": "not_evaluated",
            "novelty_against_literature": "unverified",
            "recommendation_utility_harm": "not_identified",
            "research_claim": "development-only mechanism candidates",
        },
        "warnings": [
            "A separate Qwen Critic and its descriptive SQL are not independent confirmation.",
            "Mechanism quality needs unseen matched-condition predictions; prose matching is insufficient.",
            "The statistical support of an association cannot establish recommendation-system feedback causality.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    journal = json.loads((root / "journal.json").read_text())
    state = json.loads((root / "research_state.json").read_text())
    print(json.dumps(assess(journal, state), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
