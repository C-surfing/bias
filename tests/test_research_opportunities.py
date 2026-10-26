"""The evidence pool belongs to the SAME active scientific research loop."""
from rsi_researcher.research.discovery.opportunities import ResearchOpportunityPool
import pytest


def hypothesis():
    return {
        "concept": "Unexpected conditional concentration",
        "mechanism": "recommender state changes subsequent allocation",
        "rival": "users independently change interests",
        "falsifier": "paired exposure allocation remains unchanged",
        "measurement": "paired subsequent tag concentration",
        "suggested_sql": "SELECT uid, COUNT(*) FROM events GROUP BY uid",
        "prediction_if_mechanism": "changed allocations without click changes",
        "prediction_if_rival": "changed clicks without randomized allocation shifts",
        "discriminating_experiment": "compare allocations and responses separately",
        "scientific_value": "distinguishes allocation from behavior",
        "experiment_cost": "one stratified user comparison",
    }


def experiment(estimate):
    return {"result": {"estimate": estimate, "ci95_bootstrap": [estimate-.1, estimate+.1],
                       "n_paired_units": 30},
            "specification": {"group_expression": "registered_feature"}}


def test_pool_preserves_model_mechanisms_and_numeric_countertests():
    pool = ResearchOpportunityPool()
    claim = hypothesis()
    pool.register("H001", claim, [{"sql": "SELECT COUNT(*) FROM events", "rows": [{"n": 99}]}])
    assert pool.records["H001"]["observed_evidence"][0]["rows"][0]["n"] == 99
    assert pool.records["H001"]["prediction_if_mechanism"] != pool.records["H001"]["prediction_if_rival"]
    pool.choose("H001", "Allocations differ under H1, behaviors under H2",
                "SELECT uid,COUNT(*) FROM events GROUP BY uid")
    pool.observation("H001", "SELECT ...", [{"n": 50}])
    pool.experiment("H001", experiment(.2), experiment(-.01),
                    {"decision": "test", "reason": "initial effect not robust to control"})
    assert pool.records["H001"]["tests"][0]["counter_estimate"] == -.01
    pool.update_claim("H001", claim, "abandoned")
    assert pool.focus is None
    assert pool.records["H001"]["status"] == "abandoned"
    assert pool.for_agent()["opportunities"][0]["tests"][0]["counter_users"] == 30


def test_competing_predictions_must_disagree_on_same_contrast():
    pool = ResearchOpportunityPool()
    choice = {
        "contrast_source": "events",
        "contrast_group": "measured_agent_feature",
        "contrast_outcome": "click",
        "contrast_filter": "is_random=1",
        "control_value": "0", "treatment_value": "1",
        "analysis": "paired_user", "mechanism_direction": "increase",
        "rival_direction": "no_change",
    }
    pool.check_prediction(choice)
    pool.register("H002", hypothesis(), [])
    pool.record_selection_design("H002", choice, {
        "valid_support": True, "outcome_type": "DOUBLE", "n_paired_users": 42})
    saved = pool.records["H002"]["selection_contract"]
    assert saved["group_expression"] == choice["contrast_group"]
    assert saved["development_support"]["n_paired_users"] == 42
    choice["rival_direction"] = "increase"
    with pytest.raises(ValueError, match="same measured direction"):
        pool.check_prediction(choice)


def test_rejected_opportunity_cannot_be_reselected_without_new_hypothesis():
    pool = ResearchOpportunityPool()
    pool.register("H001", hypothesis(), [])
    pool.update_claim("H001", hypothesis(), "abandoned")
    with pytest.raises(ValueError, match="abandoned"):
        pool.choose("H001", "Try same hypothesis again")
