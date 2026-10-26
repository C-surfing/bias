"""Scientific workbench tests use actual DuckDB and the real ResearchAgent loop."""
import json
import duckdb
import pytest

from rsi_researcher.research.discovery.open_research import DevelopmentTools, run_open_discovery
from rsi_researcher.research.discovery.workbench import ExperimentalWorkbench


def development_db(tmp_path):
    db = tmp_path / "isolated.duckdb"
    c = duckdb.connect(str(db))
    c.execute("""CREATE TABLE events AS SELECT
      uid, CAST(rn AS BIGINT) AS vid, 20220401 AS event_day,
      CAST(rn AS BIGINT) moment_ms,
      CAST(uid AS INTEGER)%4 AS cohort, CAST(rn AS INTEGER)%2 AS parity,
      CAST(FLOOR(rn/2)::INTEGER % 2 AS INTEGER) AS exposure,
      CASE WHEN FLOOR(rn/2)::INTEGER % 2 = 1
           THEN CASE WHEN rn%5=0 THEN 0.0 ELSE 1.0 END
           ELSE CASE WHEN rn%7=0 THEN 1.0 ELSE 0.0 END END AS click,
      0 AS is_random FROM
      range(1,23) users(uid) CROSS JOIN range(0,40) records(rn)""")
    c.execute("CREATE TABLE features AS SELECT * FROM events")
    c.close()
    return db


def spec():
    return {
        "question": "Does this dataset's arbitrary exposure code track click within users?",
        "source_table": "events",
        "source_query": "",
        "population_filter": "event_day <= 20220430",
        "unit_expression": "uid",
        "group_expression": "exposure",
        "control_value": "0", "treatment_value": "1",
        "outcome_expression": "click",
        "strata_expression": "cohort",
        "analysis": "paired_user",
        "placebo_source_query": "",
        "placebo_group_expression": "parity",
        "prediction": "exposure 1 has higher click",
        "falsifier": "no group contrast within users",
        "meaningful_margin": 0.02,
    }


def test_generic_paired_bootstrap_placebo_and_strata(tmp_path):
    db = development_db(tmp_path)
    tools = DevelopmentTools(str(db))
    try:
        result = ExperimentalWorkbench(tools).run(spec())
        assert result["result"]["n_paired_units"] == 22
        assert result["result"]["estimate"] > 0.3
        assert result["result"]["ci95_bootstrap"][0] > 0
        assert result["negative_control"] is not None
        assert abs(result["negative_control"]["result"]["estimate"]) < .2
        assert len(result["result"]["strata"]) == 4
        assert result["scientific_status"] == "development_only_not_independent_confirmation"
        assert not result["identifies_causal_bias"]
    finally:
        tools.close()


def test_source_query_and_alternative_unit(tmp_path):
    db = development_db(tmp_path)
    tools = DevelopmentTools(str(db))
    try:
        plan = spec()
        plan["source_query"] = "SELECT *, CASE WHEN exposure=1 THEN 'treated' ELSE 'control' END AS arm FROM events"
        plan["group_expression"] = "arm"
        plan["control_value"] = "control"
        plan["treatment_value"] = "treated"
        plan["placebo_group_expression"] = ""
        plan["analysis"] = "cluster_weighted"
        result = ExperimentalWorkbench(tools).run(plan)
        assert result["result"]["estimate"] > .3
        assert result["negative_control"] is None
    finally:
        tools.close()


def test_reject_invalid_design_and_insufficient_support(tmp_path):
    db = development_db(tmp_path)
    tools = DevelopmentTools(str(db))
    try:
        candidate = spec()
        candidate["group_expression"] = "nonexistent_column"
        with pytest.raises(Exception):
            ExperimentalWorkbench(tools).run(candidate)
        candidate = spec()
        candidate["population_filter"] = "1=0"
        with pytest.raises(ValueError):
            ExperimentalWorkbench(tools).run(candidate)
        candidate = spec()
        candidate["source_query"] = "SELECT * FROM read_csv('/etc/passwd')"
        with pytest.raises(ValueError):
            ExperimentalWorkbench(tools).run(candidate)
    finally:
        tools.close()


def action(name, **fields):
    item = {k: "" for k in ("reason", "query", "feature_name", "expression", "concept_id",
                           "concept", "mechanism", "rival", "falsifier", "measurement")}
    return {"action": name, "source_table": "events", **item, **fields}


def test_agent_design_executes_and_critic_requests_statistical_countertest(tmp_path, monkeypatch):
    db = development_db(tmp_path)
    import rsi_researcher.research.discovery.open_research as research
    base = spec()
    contrast = spec()
    contrast["question"] = "Check treatment effect in a different cohort"
    contrast["population_filter"] = "cohort IN (0,1)"
    claims = dict(concept="new emergent mechanism",mechanism="unknown state transition",
                  rival="user propensity",falsifier="vanishes within user",
                  measurement="event click comparison",concept_id="H001")
    decisions = [
        action("sql", reason="get summary",
               query="SELECT exposure,COUNT(*) n FROM events GROUP BY exposure"),
        action("form_concept", reason="formed from data", **claims),
        action("experiment", reason="need measured effect", concept_id="H001"),
        base,
        {"decision": "test", "reason": "Check a subgroup to distinguish user mix",
         "next_sql": "", "observable_valid": True,
         "rival_discriminated": False, "logging_artifact_excluded": False},
        contrast,
        {"decision": "revise", "reason": "subgroup result changes mechanism",
         "concept": "conditional feedback hypothesis", "mechanism": "context-specific state change",
         "rival": "user composition", "falsifier": "vanishes under within-user controls",
         "measurement": "paired interaction by cohort",
         "next_sql": "SELECT cohort, AVG(click) FROM events GROUP BY cohort"},
        action("stop", reason="insufficient causal support"),
    ]
    stages=[]
    class FakeModel:
        def ask(self,prompt,schema,context):
            stages.append(context["stage"])
            return decisions.pop(0)
    monkeypatch.setattr(research, "create_model", lambda *args: FakeModel())
    result = run_open_discovery("unused",str(tmp_path/"agent_run"),{},
                                max_rounds=4,min_queries=1,development_path=str(db))
    assert result["status"] == "development_inconclusive"
    assert len(result["concepts"][0]["tests"]) == 2
    assert result["concepts"][0]["version"] == 2
    evidence=json.loads((tmp_path/"agent_run"/"journal.json").read_text())
    stage=evidence[2]
    assert "experiment" in stage
    assert "critic_experiment" in stage
    assert stage["experiment"]["result"]["n_paired_units"] == 22
    assert "research-designer" in stages
    assert "discovery-critic" in stages
    assert not result["holdout_consulted"]

def test_model_format_normalization_is_not_a_scientific_assumption(tmp_path):
    db = development_db(tmp_path)
    tools = DevelopmentTools(str(db))
    try:
        candidate = spec()
        candidate["population_filter"] = "WHERE event_day<=20220430"
        candidate["control_value"] = "'0'"
        candidate["treatment_value"] = "'1'"
        result = ExperimentalWorkbench(tools).run(candidate)
        assert result["specification"]["control_value"] == "0"
        assert result["result"]["n_paired_units"] == 22
    finally:
        tools.close()


def test_aggregate_source_cannot_erase_unit_of_inference(tmp_path):
    db = development_db(tmp_path)
    tools = DevelopmentTools(str(db))
    try:
        candidate=spec()
        candidate["source_query"] = "SELECT exposure,AVG(click) click_rate FROM events GROUP BY exposure"
        with pytest.raises(ValueError,match="preserve per-event"):
            ExperimentalWorkbench(tools).run(candidate)
    finally:
        tools.close()


def test_placebo_cannot_be_copy_of_original_treatment(tmp_path):
    db = development_db(tmp_path)
    tools = DevelopmentTools(str(db))
    try:
        candidate=spec()
        candidate["placebo_group_expression"] = candidate["group_expression"]
        with pytest.raises(ValueError,match="negative control"):
            ExperimentalWorkbench(tools).run(candidate)
    finally:
        tools.close()
