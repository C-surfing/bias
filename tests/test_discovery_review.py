from rsi_researcher.research.discovery.review import assess


def test_review_requires_model_created_feature_numeric_countertest_and_revision():
    feature = {"derived_feature": {"name": "history_bin"}}
    experiment = {"experiment": {"result": {"estimate": .2},
                                 "specification": {"population_filter": "TRUE"}},
                  "feature_lineage": {"used": ["history_bin"]}}
    counter = {"experiment": {"result": {"estimate": .2},
                              "specification": {"population_filter": "TRUE"}},
               "critic_experiment": {"result": {"estimate": 0},
                                     "specification": {"population_filter": "cohort = 1"}},
               "revised_concept": {"version": 2},
               "reflection": {"decision": "revise"}}
    claim = {"id": "H001", "status": "investigating", "version": 2,
             "claim": {"mechanism": "effect depends on history",
                       "rival": "group composition",
                       "falsifier": "no within-user effect",
                       "measurement": "paired click difference"}}
    state = {"concepts": [claim]}
    assert assess([feature, experiment, counter], state)["full_qwen_workflow_accepted"]
    assert not assess([feature, experiment, {
        "critic_sql_countertest": {"rows": []},
        "reflection": {"decision": "revise"},
        "revised_concept": {"version": 2}}], state)["full_qwen_workflow_accepted"]
    assert not assess([experiment, counter], state)["full_qwen_workflow_accepted"]
    outcome = assess([feature, experiment, counter], state)
    assert outcome["scientific_quality"]["hidden_condition_consequence_agreement"] == "not_evaluated"


def test_review_cannot_credit_cosmetic_countertest_with_identical_numerics():
    f = {"derived_feature": {"name": "measured_arm"}}
    original = {"estimate": 0.21, "n_paired_units": 30,
                "ci95_bootstrap": [0.1, 0.3],
                "mean_control": 0.2, "mean_treatment": 0.41,
                "n_control": 300, "n_treatment": 300}
    copied = {
        "experiment": {"result": original,
                       "specification": {"population_filter": "TRUE"}},
        "critic_experiment": {"result": dict(original),
                              "specification": {"population_filter": "1=1"}},
        "feature_lineage": {"used": ["measured_arm"]},
        "reflection": {"decision": "abandon"}}
    result = assess([f, copied], {"concepts": []})
    assert not result["checks"]["critic_executed_distinct_workbench_test"]
    assert not result["full_qwen_workflow_accepted"]
