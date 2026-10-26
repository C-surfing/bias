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
    critic_workbench = [
        event for event in journal if "critic_experiment" in event
        and "experiment" in event
        and any(event["experiment"]["specification"].get(k)
                != event["critic_experiment"]["specification"].get(k)
                for k in material_keys)]
    critic_sql = [event for event in journal
                  if "critic_sql_countertest" in event]
    revised = [event for event in journal if "revised_concept" in event
               and "reflection" in event]
    abandoned = [event for event in journal
                 if event.get("reflection", {}).get("decision") == "abandon"]
    checks = {
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
    return {
        "full_qwen_workflow_accepted": accepted,
        "checks": checks,
        "counts": {"derived_features": len(derived),
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
