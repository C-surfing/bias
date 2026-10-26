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
    contrast = dict(base)
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
        result = ExperimentalWorkbench(tools).run(candidate)
        assert result["negative_control"] is None
        assert "Identical treatment" in result["omitted_invalid_placebo"]
    finally:
        tools.close()


def test_agent_registered_derived_feature_to_statistical_countertest_and_revision(tmp_path, monkeypatch):
    """The complete model/tool contract must use the feature the model actually registered."""
    db = development_db(tmp_path)
    import rsi_researcher.research.discovery.open_research as research
    hypothesis = dict(concept="contextual response asymmetry",
                      mechanism="prior exposure state changes user response",
                      rival="user or cohort composition instead of feedback",
                      falsifier="within-user or cohort contrasts erase the association",
                      measurement="paired click difference by exposure state",
                      concept_id="H001")
    original = spec()
    original.update(group_expression="registered_arm", control_value="control",
                    treatment_value="treated", placebo_group_expression="")
    counter = dict(original)
    counter.update(question="Within two independent cohorts",
                   population_filter="cohort IN (0,1)")
    decisions = [
        action("define_feature", feature_name="registered_arm",
               expression="CASE WHEN exposure=1 THEN 'treated' ELSE 'control' END"),
        action("sql", query="SELECT registered_arm,COUNT(*) n FROM events GROUP BY 1"),
        action("form_concept", **hypothesis),
        action("experiment", concept_id="H001"),
        original,
        {"decision":"test", "reason":"Check within different cohorts",
         "next_sql":"", "observable_valid":True,
         "rival_discriminated":False, "logging_artifact_excluded":False},
        counter,
        {"decision":"revise", "reason":"The countertest motivates a narrower mechanism",
         "concept":"cohort-specific response asymmetry",
         "mechanism":"the response effect depends on cohort",
         "rival":"user composition and item difficulty",
         "falsifier":"the effect vanishes when comparing same users",
         "measurement":"cohort-stratified paired click contrast",
         "next_sql":"SELECT cohort, AVG(click) AS rate FROM events GROUP BY cohort"},
        action("stop"),
    ]
    roles=[]
    class FakeModel:
        def ask(self, prompt, schema, context):
            roles.append(context["stage"])
            return decisions.pop(0)
    monkeypatch.setattr(research, "create_model", lambda *args: FakeModel())
    result = run_open_discovery("unused", str(tmp_path/"modelled"),
                                {}, max_rounds=5, min_queries=1,
                                development_path=str(db))
    journal=json.loads((tmp_path/"modelled"/"journal.json").read_text())
    assert journal[0]["derived_feature"]["name"] == "registered_arm"
    assert journal[1]["result"]["rows"]
    assert "research_events" in journal[3]["experiment"]["executed_sql"]
    assert journal[3]["experiment"]["result"]["n_paired_units"] == 22
    assert journal[3]["critic_experiment"]["result"]["n_paired_units"] >= 5
    assert result["concepts"][0]["version"] == 2
    assert len(result["concepts"][0]["revisions"]) == 1
    assert not result["holdout_consulted"]


def test_workbench_reuses_explicit_registered_view_in_custom_source(tmp_path):
    db = development_db(tmp_path)
    tools = DevelopmentTools(str(db))
    try:
        tools.define_feature("registered_arm",
                             "CASE WHEN exposure=1 THEN 'treated' ELSE 'control' END",
                             "events")
        design = spec()
        design.update(group_expression="registered_arm", control_value="control",
                      treatment_value="treated", placebo_group_expression="")
        normal = ExperimentalWorkbench(tools).run(design)
        assert "research_events" in normal["executed_sql"]
        assert normal["result"]["estimate"] > .3
        design["source_query"] = "SELECT uid,cohort,click,registered_arm,event_day FROM events"
        selected = ExperimentalWorkbench(tools).run(design)
        assert "research_events" in selected["executed_sql"]
        assert selected["result"]["estimate"] == pytest.approx(normal["result"]["estimate"])
    finally:
        tools.close()


def test_qwen_style_countertest_delta_after_copy_reuses_registered_feature(tmp_path, monkeypatch):
    """A copied Qwen countertest gets one agent-chosen difference, then numeric reflection."""
    db = development_db(tmp_path)
    import rsi_researcher.research.discovery.open_research as research
    from rsi_researcher.research.discovery.review import assess
    claim = dict(concept_id="H001", concept="feedback-context interaction",
                 mechanism="prior state changes response",
                 rival="heterogeneous users instead of feedback",
                 falsifier="user-matched effect disappears",
                 measurement="paired response by derived feedback state")
    initial = spec()
    initial.update(group_expression="registered_arm", control_value="control",
                   treatment_value="treated", placebo_group_expression="")
    fake = [
        action("define_feature", feature_name="registered_arm",
               expression="CASE WHEN exposure=1 THEN 'treated' ELSE 'control' END"),
        action("sql", query="SELECT registered_arm,COUNT(*) n FROM events GROUP BY 1"),
        action("form_concept", **claim),
        action("experiment", concept_id="H001"),
        initial,
        {"decision": "test", "reason": "Distinguish user context mix",
         "next_sql": "", "observable_valid": True,
         "rival_discriminated": False, "logging_artifact_excluded": False},
        dict(initial),
        {"field": "population_filter", "replacement": "cohort IN (0,1)",
         "reason": "Compare a narrower cohort to probe an alternative explanation"},
        {"decision": "abandon", "reason": "The alternative is not excluded by the countertest",
         "concept": claim["concept"], "mechanism": claim["mechanism"],
         "rival": claim["rival"], "falsifier": claim["falsifier"],
         "measurement": claim["measurement"], "next_sql": ""},
        action("stop"),
    ]
    class FakeModel:
        def ask(self, prompt, schema, context):
            return fake.pop(0)
    monkeypatch.setattr(research, "create_model", lambda *args: FakeModel())
    dest = tmp_path / "delta-run"
    report = run_open_discovery("unused", str(dest), {}, max_rounds=5,
                                min_queries=1, development_path=str(db))
    journal = json.loads((dest/"journal.json").read_text())
    step = next(e for e in journal if e.get("experiment"))
    assert step["feature_lineage"]["used"] == ["registered_arm"]
    assert step["critic_experiment"]["result"]["n_paired_units"] >= 5
    assert step["critic_experiment"]["specification"]["population_filter"] == "cohort IN (0,1)"
    assert step["countertest_delta"]["field"] == "population_filter"
    assert report["concepts"][0]["status"] == "rejected_development"
    assert assess(journal, report)["full_qwen_workflow_accepted"]
    assert not report["holdout_consulted"]


def test_registered_feature_name_is_required_to_bind_experiment(tmp_path):
    db = development_db(tmp_path)
    tools = DevelopmentTools(str(db))
    try:
        tools.define_feature(
            "registered_arm",
            "CASE WHEN exposure=1 THEN 'treated' ELSE 'control' END",
            "events")
        plan = spec()
        plan.update(group_expression="CASE WHEN exposure=1 THEN 'treated' ELSE 'control' END",
                    control_value="control", treatment_value="treated",
                    placebo_group_expression="")
        with pytest.raises(ValueError, match="registered feature name"):
            ExperimentalWorkbench(tools).run(plan, required_feature="registered_arm")
        plan["group_expression"] = "registered_arm"
        evidence = ExperimentalWorkbench(tools).run(plan, required_feature="registered_arm")
        assert "research_events" in evidence["executed_sql"]
        assert evidence["result"]["n_paired_units"] == 22
        card = tools.feature_cards()[0]
        assert card["name"] == "registered_arm"
        assert card["sample_distribution"][0]["n"] > 0
    finally:
        tools.close()


def test_preflight_rejects_categorical_outcome_before_expensive_inference(tmp_path):
    db = development_db(tmp_path)
    import duckdb
    con = duckdb.connect(str(db))
    con.execute("ALTER TABLE events ADD COLUMN next_tags VARCHAR DEFAULT '39,43'")
    con.close()
    tools = DevelopmentTools(str(db))
    try:
        candidate = spec()
        candidate["outcome_expression"] = "next_tags"
        with pytest.raises(ValueError, match="not a numeric per-event outcome"):
            ExperimentalWorkbench(tools).run(candidate)
        candidate["outcome_expression"] = "click"
        checked = ExperimentalWorkbench(tools).inspect_plan(candidate)
        assert checked["outcome_numeric"] and checked["valid_support"]
        assert checked["n_paired_users"] == 22
        assert checked["n_control_users"] == 22
    finally:
        tools.close()


def test_unpaired_cluster_weighted_design_is_valid_with_independent_users(tmp_path):
    db = development_db(tmp_path)
    tools = DevelopmentTools(str(db))
    try:
        candidate = spec()
        candidate.update(group_expression="uid%2", control_value="0",
                         treatment_value="1", analysis="cluster_weighted",
                         placebo_group_expression="")
        plan = ExperimentalWorkbench(tools).inspect_plan(candidate)
        assert plan["n_paired_users"] == 0
        assert min(plan["n_control_users"], plan["n_treatment_users"]) >= 5
        assert plan["valid_support"]
        measured = ExperimentalWorkbench(tools).run(candidate)["result"]
        assert measured["n_paired_units"] == 0
        assert measured["bootstrap_replicates"] >= 100
    finally:
        tools.close()


def test_strata_options_derive_from_actual_observed_cardinalities(tmp_path):
    db = development_db(tmp_path)
    tools = DevelopmentTools(str(db))
    try:
        candidates = ExperimentalWorkbench(tools).observed_strata_columns("events")
        assert "" in candidates
        assert "cohort" in candidates and "exposure" in candidates
        # High-cardinality event timestamps and outcome-defined strata are
        # not accepted as the first study's conditioning variables.
        assert "moment_ms" not in candidates
        assert "click" not in candidates
    finally:
        tools.close()


def test_selection_with_zero_observed_control_users_is_rejected(tmp_path):
    db = development_db(tmp_path)
    tools = DevelopmentTools(str(db))
    try:
        candidate = spec()
        candidate.update(group_expression="exposure",
                         population_filter="exposure=1",
                         placebo_group_expression="")
        checked = ExperimentalWorkbench(tools).inspect_plan(candidate)
        assert checked["n_control_users"] == 0
        assert not checked["valid_support"]
        with pytest.raises(ValueError, match="independent user support"):
            ExperimentalWorkbench(tools).run(candidate)
    finally:
        tools.close()
