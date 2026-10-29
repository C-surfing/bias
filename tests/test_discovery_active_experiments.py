"""Longitudinal test: open mechanisms can be explored without six parameters."""
import duckdb
import pytest

from rsi_researcher.research.discovery.active_experiments import (
    assess_candidates, available_capabilities, choose_feasible,
    execute_observation,
)
from rsi_researcher.research.discovery.datasets import KuaiRandAdapter
from rsi_researcher.research.discovery.open_research import DevelopmentTools


CROSS_USER_SQL = """
WITH prior_other AS (
    SELECT later.uid, later.vid, later.moment_ms, later.click,
           COUNT(earlier.uid) FILTER (
               WHERE earlier.uid <> later.uid
                 AND earlier.moment_ms < later.moment_ms
                 AND earlier.click = 1
           ) AS past_other_clicks
    FROM events AS later
    LEFT JOIN events AS earlier
      ON earlier.vid = later.vid
     AND earlier.moment_ms < later.moment_ms
     AND earlier.uid <> later.uid
    GROUP BY later.uid, later.vid, later.moment_ms, later.click
)
SELECT CASE WHEN past_other_clicks > 0 THEN 1 ELSE 0 END AS prior_other_positive,
       COUNT(*) n_events, COUNT(DISTINCT uid) n_users,
       AVG(click) AS later_click_rate
FROM prior_other
GROUP BY 1
ORDER BY 1
"""


def proposed(kind, query, h1="increase", h2="decrease"):
    return {
        "concept_id": "H001",
        "question": "Does earlier feedback from other users precede later responses?",
        "required_capability": kind,
        "study_unit": "later user-video-event",
        "target_quantity": "click response after earlier other-user feedback",
        "prediction_h1": "conditional response increases",
        "prediction_h2": "conditional response decreases",
        "direction_h1": h1, "direction_h2": h2,
        "sql": query, "reference_assumption": "no normative policy reference",
        "mechanism_link": "other-user feedback precedes target-user exposure",
    }


@pytest.fixture
def dev_tools(tmp_path):
    path = str(tmp_path / "dev.duckdb")
    con = duckdb.connect(path)
    con.execute(
        "CREATE TABLE events(uid INT, vid INT, moment_ms BIGINT, "
        "event_day INT, is_random INT, click INT)")
    # Earlier feedback from user u=1; later independent users react to
    # the same item. This is a feasibility fixture, NOT an identified Bias.
    records = [(uid, vid, uid * 10 + vid, 20220423,
                uid % 2, (uid + vid) % 2)
               for vid in range(1, 6) for uid in range(1, 7)]
    con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", records)
    con.execute("CREATE TABLE features AS SELECT * FROM events")
    con.close()
    tools = DevelopmentTools(path, adapter=KuaiRandAdapter())
    try:
        yield tools
    finally:
        tools.close()


def test_cross_user_temporal_observation_executes_without_controlled_parameter(dev_tools):
    concepts = [{"id": "H001", "status": "investigating"}]
    plans = [
        proposed("sql_observation", CROSS_USER_SQL),
        proposed("controlled_parameter", "", "increase", "no_change"),
        proposed("sequential_intervention", "", "increase", "decrease"),
    ]
    assert available_capabilities(dev_tools.adapter)["controlled_parameter"]["available"] is False
    checked = assess_candidates(dev_tools, concepts, plans)
    assert checked[0]["status"] == "feasible_descriptive_observation"
    assert checked[1]["status"] == "not_executable_with_current_capabilities"
    assert checked[2]["status"] == "not_executable_with_current_capabilities"
    # EXPLAIN preflight must not run the actual observation.
    assert dev_tools.history == []
    chosen = choose_feasible(checked)
    evidence = execute_observation(dev_tools, chosen)
    assert evidence["concept_id"] == "H001"
    assert len(evidence["executed_evidence"]["rows"]) == 2
    assert evidence["h1_h2_discriminated"] is False
    assert evidence["reference_identified"] is False
    assert evidence["production_algorithm_causality_identified"] is False
    assert len(dev_tools.history) == 1
    assert assess_candidates(dev_tools, concepts, [plans[0]])[0]["status"] == "already_measured"


def test_prediction_disagreement_does_not_override_execution_support(dev_tools):
    concepts = [{"id": "H001", "status": "investigating"}]
    same = proposed("sql_observation", CROSS_USER_SQL, "increase", "increase")
    invalid = proposed("sql_observation", "SELECT nonexistent FROM events")
    unsupported = proposed("policy_comparison", "")
    result = assess_candidates(dev_tools, concepts, [same, invalid, unsupported])
    assert [r["status"] for r in result] == [
        "not_discriminating", "invalid_sql",
        "not_executable_with_current_capabilities"]
    assert choose_feasible(result) is None
    assert dev_tools.history == []


def test_bounded_sql_gateway_still_rejects_unsafe_queries(dev_tools):
    with pytest.raises(ValueError, match="bounded SELECT"):
        dev_tools.preflight_sql("DROP TABLE events")
    with pytest.raises(ValueError, match="bounded SELECT"):
        dev_tools.preflight_sql("SELECT * FROM events; SELECT * FROM features")
    assert dev_tools.history == []
