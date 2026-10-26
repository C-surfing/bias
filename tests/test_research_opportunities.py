"""The evidence pool belongs to the SAME active scientific research loop."""
from rsi_researcher.research.discovery.opportunities import ResearchOpportunityPool


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
