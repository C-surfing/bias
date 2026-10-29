"""Deterministic handoff between selected research contrasts and execution.

The agent owns which mechanism/contrast to test; the harness must not ask a
second agent to reinvent an already prevalidated primary estimand.
"""
from __future__ import annotations


def previously_measured(query: str, history: list[dict]) -> dict | None:
    """Return existing exact SQL evidence without producing another tool call."""
    requested = query.strip().rstrip(";").strip()
    for result in reversed(history):
        if result.get("sql", "").strip().rstrip(";").strip() == requested:
            return result
    return None


def pending_primary_contract(concepts: list[dict], opportunities: dict) -> str | None:
    """Pick one active, preregistered but not yet executed primary comparison."""
    for concept in concepts:
        if concept["status"] in ("rejected_development", "design_blocked", "frozen_unconfirmed"):
            continue
        contract = opportunities.get(concept["id"], {}).get("selection_contract")
        if not contract or contract.get("concept_version", concept["version"]) != concept["version"]:
            continue
        if not any("workbench" in trial and trial.get("version") == concept["version"]
                   for trial in concept["tests"]):
            return concept["id"]
    return None


def primary_experiment_intent(contract: dict, concept: dict) -> dict:
    """Compile the ALREADY supported H1/H2 contrast, not another model guess.

    This is an observational development test. It cannot alone identify a
    production ranking mechanism or exclude a preference-based rival.
    """
    estimand = contract["estimand_spec"]
    return {
        "question": "Preregistered H1/H2 development contrast: "
                    + concept["claim"]["measurement"],
        "group_feature": estimand["group_expression"],
        "control_value": estimand["control_value"],
        "treatment_value": estimand["treatment_value"],
        "outcome": estimand["outcome_expression"],
        "population_filter": estimand["population_filter"],
        "analysis": estimand["analysis"],
        "prediction": "H1: " + contract["prediction_if_mechanism"],
        "falsifier": "H2: " + contract["prediction_if_rival"],
        "meaningful_margin": 0.01,
    }


def audit_design_role(specification: dict, selection_contract: dict | None,
                      *, primary: bool) -> dict:
    """Preserve primary estimand; permit explicitly distinct inferential tests.

    The secondary experiment belongs to the same hypothesis version but is
    not a substitute for the registered main comparison and cannot silently
    be used as evidence about the original estimand.
    """
    if not selection_contract:
        return {"purpose": "exploratory", "parent_estimand": None}
    from rsi_researcher.research.discovery.references import EstimandSpec
    parent = selection_contract["estimand_spec"]
    comparison = EstimandSpec(**parent).compare_executed(specification)
    same = comparison["same_estimand"]
    if primary and not same:
        raise ValueError(
            "Primary experiment changed preregistered estimand; use "
            "the selected contract, not a new comparison")
    if not primary and same:
        raise ValueError(
            "Repeated primary estimand is not a rival-discriminating "
            "countertest; change the condition, outcome or study design")
    return {
        "purpose": "preregistered_primary" if primary else "distinct_countertest",
        "parent_estimand": parent, "matches_original_contrast": same,
        **comparison,
        "countertest_not_primary_substitute": not primary,
    }
