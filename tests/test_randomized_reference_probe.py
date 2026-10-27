import duckdb
import pytest
from rsi_researcher.research.discovery.open_research import DevelopmentTools
from rsi_researcher.research.discovery.workbench import randomized_reference_probe
from rsi_researcher.research.discovery.opportunities import ResearchOpportunityPool


def test_numeric_reference_has_actual_overlapping_days_and_no_causal_promotion(tmp_path):
    path = str(tmp_path / "reference.duckdb")
    c = duckdb.connect(path)
    c.execute("CREATE TABLE events(uid INT,vid INT,event_day INT,is_random INT,previous_click INT,click INT)")
    rows = []
    for uid in range(1, 25):
        for policy in (0, 1):
            for arm, n in ((0, 8 if policy == 0 else 5),
                           (1, 2 if policy == 0 else 5)):
                for i in range(n):
                    rows.append((uid, i, 20220423, policy, arm, arm if policy else 0))
        # This non-overlapping earlier standard-only day must be excluded.
        rows.extend((uid, i, 20220410, 0, 1, 1) for i in range(40))
    c.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", rows)
    c.execute("CREATE TABLE features AS SELECT * FROM events LIMIT 0")
    c.close()
    tools = DevelopmentTools(path)
    choice = dict(contrast_source="events", reference_kind="randomized_exposure",
                  contrast_group="previous_click", contrast_outcome="click",
                  contrast_filter="", control_value="0", treatment_value="1")
    try:
        measured = randomized_reference_probe(tools, choice)
    finally:
        tools.close()
    assert measured["same_calendar_days"] == [20220423]
    assert measured["random_response_contrast_treatment_minus_control"] == pytest.approx(1.0)
    assert measured["standard_minus_random_treatment_group_share"] == pytest.approx(-.3)
    assert measured["n_random_users_by_arm"] == {"0": 24, "1": 24}
    assert measured["response_vs_composition_sign_disagreement"]
    assert measured["standard_response_contrast_treatment_minus_control"] == pytest.approx(0.0)
    assert measured["standard_minus_random_response_contrast"] == pytest.approx(-1.0)
    assert measured["policy_response_interaction_ci95_user_bootstrap"] == pytest.approx([-1.0, -1.0])
    assert not measured["discrepancy_is_validated_bias"]
    pool = ResearchOpportunityPool()
    pool.register("H001", dict(concept="test", mechanism="H1", rival="H2",
                               falsifier="F", measurement="M", suggested_sql="SELECT 1"), [])
    pool.record_reference_diagnostic("H001", measured)
    assert pool.records["H001"]["evidence_status"] == "reference_diagnostic_only"
    assert pool.records["H001"]["bias_relevance"]["reference_measured"] is False
    audit = pool.records["H001"]["bias_relevance"]["numeric_discrepancy"]
    assert audit["observed"] == pytest.approx(.2)
    assert audit["reference"] == pytest.approx(.5)
    assert audit["estimate"] == pytest.approx(audit["observed"] - audit["reference"])
    assert audit["ci95"] == pytest.approx([-.3, -.3])
    assert audit["measured"] and not audit["normative_target_validated"]


def test_derived_feature_cannot_use_future_information_or_unobserved_arms(tmp_path):
    path = str(tmp_path / "features.duckdb")
    conn = duckdb.connect(path)
    conn.execute("CREATE TABLE events(uid INT, vid INT, event_day INT, tags VARCHAR,"
                 " next_tags VARCHAR, click INT, is_random INT)")
    conn.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)", [
        (i, i, 20220423, str(i % 12), str((i + 1) % 12), 0, i % 2)
        for i in range(48)])
    conn.execute("CREATE TABLE features AS SELECT * FROM events LIMIT 0")
    conn.close()
    tools = DevelopmentTools(path)
    try:
        with pytest.raises(ValueError, match="future"):
            tools.define_feature("future_group", "next_tags", "events")
        with pytest.raises(ValueError, match="2-8"):
            tools.define_feature("raw_id_group", "tags", "events")
        eligible = tools.define_feature(
            "item_bin", "CASE WHEN tags = '3' THEN 1 ELSE 0 END", "events")
        assert {r["item_bin"] for r in eligible["evidence"]["rows"]} == {0, 1}
    finally:
        tools.close()
