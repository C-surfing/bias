"""Scientific state regressions for open discovery; no model service or holdout."""
import duckdb

from rsi_researcher.research.discovery.open_research import (
    DevelopmentTools, scientific_evidence_gaps,
)
from rsi_researcher.research.discovery.opportunities import ResearchOpportunityPool
from rsi_researcher.research.discovery.workbench import ExperimentalWorkbench
from rsi_researcher.research.discovery.review import assess


def _pool():
    pool = ResearchOpportunityPool()
    pool.register("H001", {
        "concept": "candidate allocation discrepancy",
        "mechanism": "model allocation changes",
        "rival": "user preference explains the change",
        "falsifier": "difference disappears under common context",
        "measurement": "paired exposure share",
        "suggested_sql": "SELECT 1",
    }, [])
    return pool


def test_reference_measurement_is_recorded_but_not_normative():
    pool = _pool()
    pool.record_reference_measurement("H001", {
        "estimate": 0.12, "ci95_bootstrap": [0.04, 0.2],
        "mean_control": 0.4, "mean_treatment": 0.55,
    }, scope="matched_item_pool_standard_minus_random_diagnostic")
    bias = pool.records["H001"]["bias_relevance"]
    assert bias["reference_measured"]
    assert bias["numeric_discrepancy"]["estimate"] == 0.12
    assert not bias["numeric_discrepancy"]["normative_target_validated"]
    assert "normative_reference_not_identified" in scientific_evidence_gaps(
        {"tests": [], "critique": None}, pool.records["H001"])


def test_consequence_gap_only_closes_on_actual_consequence_evidence():
    pool = _pool()
    concept = {"tests": [], "critique": None}
    record = pool.records["H001"]
    assert "recommendation_consequence_not_identified" in scientific_evidence_gaps(concept, record)
    record["recommendation_consequence"] = {"identified": True, "evidence_id": "T001"}
    assert "recommendation_consequence_not_identified" not in scientific_evidence_gaps(
        concept, record)


def test_native_contrast_not_forced_to_reuse_unrelated_derived_feature(tmp_path):
    path = str(tmp_path / "dev.duckdb")
    conn = duckdb.connect(path)
    conn.execute("CREATE TABLE events AS SELECT i // 20 uid, i vid,"
                 " 20220423 event_day, i % 2 native_arm, i % 3 click,"
                 " i % 5 feature_seed FROM range(400) t(i)")
    conn.execute("CREATE TABLE features AS SELECT uid, vid, click FROM events")
    conn.close()
    tools = DevelopmentTools(path)
    try:
        tools.define_feature("unrelated_bin",
                             "CASE WHEN feature_seed > 2 THEN 1 ELSE 0 END",
                             source_table="events")
        spec = {
            "question": "Do original native groups differ?",
            "source_table": "events", "source_query": "",
            "population_filter": "", "unit_expression": "uid",
            "group_expression": "native_arm",
            "control_value": "0", "treatment_value": "1",
            "outcome_expression": "click", "strata_expression": "",
            "analysis": "paired_user", "placebo_source_query": "",
            "placebo_group_expression": "",
            "prediction": "H1 predicts a group difference",
            "falsifier": "H2 predicts no difference",
            "meaningful_margin": 0.01,
        }
        measured = ExperimentalWorkbench(tools).run(spec, required_feature=None)
        assert measured["result"]["n_paired_units"] == 20
    finally:
        tools.close()


def test_review_native_contrast_without_derived_feature_is_valid_workflow_only():
    spec = {"source_query": "", "population_filter": "",
            "group_expression": "native_arm", "outcome_expression": "click",
            "control_value": "0", "treatment_value": "1"}
    first = {"estimate": 0.2, "mean_control": 0.3, "mean_treatment": 0.5,
             "n_control": 200, "n_treatment": 200, "n_paired_units": 20,
             "ci95_bootstrap": [0.1, 0.3]}
    second = {**first, "estimate": 0.1, "ci95_bootstrap": [0.01, 0.2]}
    journal = [
        {"opportunity_selection": {
            "selected_concept_id": "H001", "mechanism_direction": "increase",
            "rival_direction": "no_change"},
         "selection_feasibility": {"valid_support": True, "outcome_numeric": True}},
        {"experiment": {"result": first, "specification": spec},
         "selection_contract_audit": {"matches_original_contrast": True},
         "critic_experiment": {"result": second, "specification": {
             **spec, "population_filter": "feature_seed > 2"}},
         "reflection": {"decision": "abandon"}},
    ]
    result = assess(journal, {"concepts": []})
    assert result["full_qwen_workflow_accepted"]
    assert result["scientific_quality"]["bias_reference_discrepancy_verified"] is False
    assert result["scientific_quality"]["independently_confirmed_recommender_bias"] is False
