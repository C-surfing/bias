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


def test_full_agent_route_active_probe_then_primary_and_countertest(tmp_path, monkeypatch):
    """One research cycle reaches real Workbench H1/H2 without fixed parameters.

    The LLM is replaced by a deterministic scripted scientist; SQL, workbench,
    model role handoffs, evidence records and reference gates all run for real.
    """
    import json
    from rsi_researcher.research.discovery import observations
    from rsi_researcher.research.discovery import open_research as research

    path = str(tmp_path / "route.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE events(uid INT,vid INT,event_day INT,"
                "moment_ms BIGINT,is_random INT,click INT,previous_click INT)")
    con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?)", [
        (uid, vid, 20220423, uid * 1000 + vid, uid % 2,
         (uid + vid) % 2, vid % 2)
        for uid in range(1, 41) for vid in range(1, 7)])
    con.execute("CREATE TABLE features AS SELECT * FROM events")
    con.close()

    cards = [
        {"observation_id": "E001", "source_table": "events",
         "column": "previous_click", "outcome": "click",
         "operation": "generic_conditional_mean",
         "sample_n": 240, "measured_rows": [
             {"value": "0", "mean": 0.5}, {"value": "1", "mean": 0.5}]},
        {"observation_id": "E002", "source_table": "events",
         "column": "is_random", "outcome": "click",
         "operation": "generic_conditional_mean",
         "sample_n": 240, "measured_rows": [
             {"value": "0", "mean": 0.5}, {"value": "1", "mean": 0.5}]},
    ]
    monkeypatch.setattr(observations, "build_exploration_evidence",
                        lambda tools, profile: cards)
    base = {
        "mechanism": "cross-user feedback affects subsequent rank allocation",
        "rival": "stable video relevance explains the observed association",
        "falsifier": "the association vanishes after item/time matching",
        "measurement": "conditional later response",
        "suggested_sql": "SELECT previous_click,COUNT(*) n FROM events GROUP BY 1",
        "target_reference": "unmeasured",
        "reference_identification": "requires controlled platform allocation",
        "distortion_quantity": "feedback-conditioned opportunity",
        "distortion_estimate": "unknown",
        "prediction_if_mechanism": "increase",
        "prediction_if_rival": "decrease",
        "affected_population": "same-video later users",
        "mechanism_entry_point": "item ranking feedback",
        "reference_status": "unidentified",
        "next_information_needed": "controlled exposure policy",
        "discriminating_experiment": "temporally ordered other-user feedback",
    }
    chooser = {
        "selected_concept_id": "H001", "reason": "test one observed contrast",
        "discriminating_prediction": "earlier-versus-later signal",
        "prediction_if_mechanism": "higher later response",
        "prediction_if_rival": "lower later response",
        "why_not_other": "paired users available",
        "next_action": "experiment", "next_sql": "",
        "experimental_cost": "small",
        "contrast_source": "events", "contrast_group": "previous_click",
        "contrast_outcome": "click", "contrast_filter": "",
        "control_value": "0", "treatment_value": "1", "analysis": "paired_user",
        "mechanism_direction": "increase", "rival_direction": "decrease",
        "reference_kind": "none", "target_reference": "unmeasured",
        "proposed_distortion": "possible allocation difference",
        "mechanism_entry_point": "ranking feedback",
        "corrective_intervention": "undetermined",
    }
    stages = []

    class ScriptedModel:
        def ask(self, prompt, schema, context):
            # ResearchAgent puts the role in context, the specific call
            # episode in model.scope; use the latter to distinguish Designer tasks.
            stage = self.scope["episode"]
            stages.append(stage)
            if stage.startswith("phenomenon-extraction-"):
                return {"phenomena": [
                    {"description": "previous_click conditional response pattern",
                     "evidence_ids": ["E001"],
                     "conditions": "grouped by previous_click",
                     "observed_pattern": "observed conditional response difference",
                     "alternative": "latent preference", "next_measurement": "cross-user SQL"},
                    {"description": "is_random conditional response pattern",
                     "evidence_ids": ["E002"],
                     "conditions": "grouped by is_random",
                     "observed_pattern": "observed conditional response difference",
                     "alternative": "item supply", "next_measurement": "reference comparison"},
                ]}
            if stage.startswith("concept-formation-"):
                return {"concepts": [
                    {**base, "concept": "cross-user temporal feedback association",
                     "phenomenon_ids": ["P001"]},
                    {**base, "concept": "mode-conditioned exposure opportunity",
                     "measurement": "conditional mode response",
                     "phenomenon_ids": ["P002"]},
                ]}
            if stage.startswith("active-experiment-design-"):
                return {"candidates": [
                    proposed("sql_observation", CROSS_USER_SQL),
                    {**proposed("sequential_intervention", ""),
                     "concept_id": "H002"},
                ]}
            if stage.startswith("opportunity-choice-"):
                return chooser
            if stage.startswith("experiment-critique-"):
                return {"decision": "test", "reason": "preference rival unresolved",
                        "next_sql": "", "observable_valid": True,
                        "rival_discriminated": False,
                        "logging_artifact_excluded": False}
            if stage.startswith("experiment-countertest-"):
                return {
                    "question": "Conditional user preference sensitivity test",
                    "group_feature": "previous_click",
                    "control_value": "0", "treatment_value": "1",
                    "outcome": "click",
                    "population_filter": "(TRUE) AND is_random = 0",
                    "analysis": "paired_user",
                    "prediction": "H1 increase", "falsifier": "H2 decrease",
                    "meaningful_margin": 0.01,
                }
            if stage.startswith("experiment-post-counter-critique-"):
                return {"decision": "test", "reason": "still observational",
                        "next_sql": "", "observable_valid": True,
                        "rival_discriminated": False,
                        "logging_artifact_excluded": False}
            if stage.startswith("experiment-reflection-"):
                return {
                    "decision": "abandon",
                    "reason": "without policy intervention the mechanism remains unidentified",
                    "concept": "cross-user temporal feedback association",
                    "mechanism": base["mechanism"], "rival": base["rival"],
                    "falsifier": base["falsifier"],
                    "measurement": base["measurement"], "next_sql": "",
                }
            if stage.startswith("open-"):
                return {
                    "action": "stop", "reason": "Fixture completed its rival test",
                    "query": "", "feature_name": "", "expression": "",
                    "source_table": "events", "concept_id": "",
                    "concept": "", "mechanism": "", "rival": "",
                    "falsifier": "", "measurement": "",
                }
            raise AssertionError("unexpected model call: " + stage)

    monkeypatch.setattr(research, "create_model",
                        lambda *args, **kwargs: ScriptedModel())
    output = tmp_path / "research"
    result = research.run_open_discovery(
        "unused", str(output),
        {"llm": {"provider": "vllm"},
         "discovery": {"open_phenomena": True,
                       "active_experiment_planning": True}},
        max_rounds=3, min_queries=0, development_path=path)
    journal = json.loads((output / "journal.json").read_text())
    active = [item for item in journal if "active_experiment" in item]
    assert active, "A chosen unsupported-parameter-free SQL should execute"
    assert active[0]["active_experiment"]["h1_h2_discriminated"] is False
    assert any(item["status"] == "not_executable_with_current_capabilities"
               for item in active[0]["candidate_experiment_audit"])
    primary = [item for item in journal if "experiment" in item]
    assert primary, "Preregistered primary Workbench test should execute"
    assert primary[0]["primary_experiment_source"] == "selection_contract"
    assert primary[0]["experiment"]["identifies_causal_bias"] is False
    assert "critic_experiment" in primary[0], "Countertest should execute"
    assert "reflection" in primary[0]
    assert result["independent_confirmation"] is False
    assert any(s.startswith("active-experiment-design-") for s in stages)
    assert not any(s.startswith("experiment-design-") for s in stages)
