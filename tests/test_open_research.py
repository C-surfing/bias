"""Real code path checks: open research tools, role handoff, concept revision."""
import json

import duckdb
import pytest

from rsi_researcher.research.discovery import open_research as o


def _raw(tmp_path):
    source = tmp_path / "source.duckdb"
    db = duckdb.connect(str(source))
    db.execute("""CREATE TABLE anomaly_features AS
        SELECT * FROM (VALUES
          (1, 101, 20220420, 1000, 'standard', 'active', '3',
           1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 60000, .8, .6, 1),
          (1, 102, 20220421, 2000, 'standard', 'active', '4',
           0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 8000000, .2, .5, 1),
          (2, 201, 20220506, 3000, 'standard', 'active', '3',
           0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 60000, .2, .5, 0)
        ) AS t(uid,vid,event_day,moment_ms,source_file,activity,tags,
               click,long_view,next_click,after_long_gap,after_nonclick,
               after_negative_feedback,immediate_repeat,topic_changed,
               recent_response_drop,after_other_users_negative_response,
               previous_interval_ms,recent_click,older_click,user_fold)""")
    db.execute("""CREATE TABLE observed_sequences AS SELECT
        uid,vid,event_day,moment_ms,0 AS is_random,click,0 AS hate,
        long_view,tags,0 AS previous_video,0 AS previous_click,
        0 AS previous_hate,0 AS previous_moment,'3' AS previous_tags,
        '3' AS next_tags FROM anomaly_features""")
    db.close()
    return source


def _act(action, **values):
    obj = dict(action=action, reason="investigate", query="", feature_name="",
               expression="", source_table="features", concept_id="", concept="", mechanism="",
               rival="", falsifier="", measurement="")
    obj.update(values)
    return obj


def test_dev_isolation_sql_and_trajectory(tmp_path):
    source = _raw(tmp_path)
    dest = tmp_path / "dev.duckdb"
    o.development_db(str(source), str(dest))
    tools = o.DevelopmentTools(str(dest))
    try:
        assert tools.sql("SELECT count(*) n FROM features")["rows"][0]["n"] == 2
        assert tools.sql("SELECT count(*) n FROM events")["rows"][0]["n"] == 2
        assert "after_nonclick" in [c["column"] for c in tools.schema()["features"]]
        trace = tools.inspect_trajectory("SELECT uid FROM events GROUP BY uid ORDER BY uid LIMIT 1")
        assert trace["selected_uid"] == 1
        assert len(trace["trajectory"]["rows"]) == 2
        for statement in ("SELECT * FROM read_csv('/etc/passwd')",
                          "ATTACH '/tmp/foo.duckdb' AS a",
                          "SELECT 1; SELECT 2", "CREATE TABLE hack AS SELECT 1"):
            with pytest.raises(Exception):
                tools.sql(statement)
    finally:
        tools.close()


def test_derived_feature_is_queryable_and_ephemeral(tmp_path):
    source = _raw(tmp_path)
    dest = tmp_path / "dev.duckdb"
    o.development_db(str(source), str(dest))
    tools = o.DevelopmentTools(str(dest))
    try:
        result = tools.define_feature(
            "feedback_age_bin",
            "CASE WHEN previous_interval_ms > 7200000 THEN 'stale' ELSE 'recent' END")
        assert result["usable_table"] == "features"
        rows = tools.sql(
            "SELECT feedback_age_bin,COUNT(*) n FROM research_features GROUP BY feedback_age_bin")["rows"]
        assert {x["feedback_age_bin"]: x["n"] for x in rows} == {"stale": 1, "recent": 1}
        with pytest.raises(ValueError):
            tools.define_feature("danger", "read_csv('/etc/passwd')")
    finally:
        tools.close()
    # No persistent schema mutation is introduced by the agent.
    db = duckdb.connect(str(dest), read_only=True)
    with pytest.raises(Exception):
        db.execute("SELECT * FROM research_features")
    db.close()


def test_real_research_loop_forms_tests_revises_and_checks_literature(tmp_path, monkeypatch):
    source = _raw(tmp_path)
    dev = tmp_path / "dev.duckdb"
    o.development_db(str(source), str(dev))
    q = "SELECT feedback_age_bin,after_nonclick,COUNT(*) n,AVG(click) rate FROM research_features GROUP BY 1,2"
    base = dict(concept_id="H001", concept="feedback expiry mismatch",
                mechanism="previous non-click becomes less predictive across long lags",
                rival="session mix and logging artifacts",
                falsifier="the gap should disappear after controlling for user and topic",
                measurement="time x previous_nonclick interaction with sample counts")
    responses = [
        _act("inspect_trajectory", query="SELECT uid FROM events GROUP BY uid ORDER BY uid LIMIT 1"),
        _act("define_feature", feature_name="feedback_age_bin",
             expression="CASE WHEN previous_interval_ms > 7200000 THEN 'stale' ELSE 'recent' END"),
        _act("form_concept", **base),
        _act("test", query=q, **base),
        {"decision": "test", "reason": "need alternative lag contrast",
         "next_sql": "SELECT after_nonclick,AVG(click) rate FROM features GROUP BY 1",
         "observable_valid": True, "rival_discriminated": False, "logging_artifact_excluded": False},
        {"decision": "revise", "reason": "need a different test",
         "concept": "feedback time mismatch", "mechanism": "context-specific carryover",
         "rival": "selection", "falsifier": "user matched effect vanishes",
         "measurement": "elapsed-time by feedback interaction",
         "next_sql": "SELECT after_long_gap,COUNT(*) n FROM features GROUP BY 1"},
        _act("revise", reason="new evidence suggests a weak lag interaction",
             **{**base, "concept": "session conditional feedback memory"}),
        _act("test", query="SELECT after_nonclick,COUNT(*) n,AVG(click) rate FROM features GROUP BY 1", **base),
        {"decision": "ready", "reason": "preliminary discriminating test",
         "next_sql": "", "observable_valid": True,
         "rival_discriminated": True, "logging_artifact_excluded": True},
        _act("literature", **base),
        {"search_queries": ["feedback temporal decay recommendation",
                            "negative feedback session shift recommender"]},
        {"overlap": "earlier work discusses lag", "unresolved_difference": "model-state expiry",
         "strongest_prior_work_ids": ["https://openalex.org/W1"],
         "verdict": "possibly_distinct",
         "discriminating_next_test": "replay session state update"},
        _act("freeze", **base),
    ]
    calls = []
    class FakeModel:
        def ask(self, prompt, schema, context):
            calls.append(context["stage"])
            return responses.pop(0)
    monkeypatch.setattr(o, "create_model", lambda *a: FakeModel())
    monkeypatch.setattr(o, "search_openalex", lambda qs: {
        "queries": qs, "papers": [
            {"id": "https://openalex.org/W1", "title": "Prior work"},
            {"id": "https://openalex.org/W2", "title": "Related work"}],
        "errors": [], "retrieval_complete": True})
    result = o.run_open_discovery(str(source), str(tmp_path / "run"), {},
                                  max_rounds=9, development_path=str(dev))
    assert result["status"] == "frozen_unconfirmed"
    assert result["frozen_id"] == "H001"
    assert result["concepts"][0]["version"] == 3
    assert len(result["concepts"][0]["tests"]) == 3
    assert result["concepts"][0]["literature"]["assessment"]["verdict"] == "possibly_distinct"
    assert result["holdout_consulted"] is False
    assert "discovery-critic" in calls
    assert "literature-researcher" in calls
    assert not (tmp_path / "run" / "holdout_assessment.json").exists()


def test_freeze_gate_rejects_untested_concept(tmp_path, monkeypatch):
    source = _raw(tmp_path)
    dev = tmp_path / "dev.duckdb"
    o.development_db(str(source), str(dev))
    proposal = dict(concept_id="H001", concept="lag response",
                    mechanism="time effect", rival="other users",
                    falsifier="difference disappears with controls",
                    measurement="average click by lag")
    outputs = [
        _act("sql", query="SELECT count(*) n FROM features"),
        _act("form_concept", **proposal),
        _act("freeze", **proposal),
        _act("stop", **proposal),
    ]
    class FakeModel:
        def ask(self, prompt, schema, context):
            return outputs.pop(0)
    monkeypatch.setattr(o, "create_model", lambda *a: FakeModel())
    result = o.run_open_discovery(str(source), str(tmp_path / "run"), {},
                                  max_rounds=4, development_path=str(dev))
    assert result["status"] == "development_inconclusive"
    journal = json.loads((tmp_path / "run" / "journal.json").read_text())
    assert journal[2]["freeze_gate"]["has_development_test"] is False


def test_literature_failure_remains_insufficient(monkeypatch):
    class FakeModel:
        def ask(self, prompt, schema, context):
            if "search_queries" in schema["properties"]:
                return {"search_queries": ["negative feedback recommender",
                                           "session effect recommendation"]}
            return {"overlap": "", "unresolved_difference": "",
                    "strongest_prior_work_ids": [],
                    "verdict": "possibly_distinct",
                    "discriminating_next_test": "further reading"}
    monkeypatch.setattr(o, "search_openalex", lambda qs: {
        "queries": qs, "papers": [], "errors": [{"error": "offline"}],
        "retrieval_complete": False})
    value = o.literature_review(FakeModel(), {"mechanism": "test"}, "ep")
    assert value["assessment"]["verdict"] == "insufficient_literature"


def test_compact_context_does_not_destroy_full_data():
    result = {"sql": "SELECT 1", "columns": ["n"],
              "rows": [{"n": i} for i in range(30)]}
    short = o.compact_query(result)
    assert len(short["rows"]) == 8
    assert short["model_excerpt_only"]
    assert len(result["rows"]) == 30

def test_event_derived_feature_uses_selected_source(tmp_path):
    source = _raw(tmp_path)
    db = tmp_path / "dev.duckdb"
    o.development_db(str(source), str(db))
    tools = o.DevelopmentTools(str(db))
    try:
        result = tools.define_feature(
            "tag_transition",
            "CASE WHEN tags=previous_tags THEN 1 ELSE 0 END", source_table="events")
        assert result["usable_table"] == "events"
        rows = tools.sql(
            "SELECT tag_transition,COUNT(*) n FROM events GROUP BY tag_transition")["rows"]
        assert sum(row["n"] for row in rows) == 2
    finally:
        tools.close()


def test_auto_concept_stage_produces_unpreselected_hypotheses(tmp_path, monkeypatch):
    source = _raw(tmp_path)
    db = tmp_path / "dev.duckdb"
    o.development_db(str(source), str(db))
    action = _act
    claims = [
        {"concept": "context-dependent feedback decay",
         "mechanism": "feedback informativeness changes after session breaks",
         "rival": "selection across sessions",
         "falsifier": "within-user contrast disappears",
         "measurement": "click difference by elapsed-time bins",
         "suggested_sql": "SELECT after_long_gap,COUNT(*) n,AVG(click) rate FROM features GROUP BY 1"},
        {"concept": "tag transition quality mismatch",
         "mechanism": "tag transitions decouple expected utility from exposures",
         "rival": "content novelty preference",
         "falsifier": "quality effect absent within users",
         "measurement": "tag change x user outcome",
         "suggested_sql": "SELECT topic_changed,COUNT(*) n,AVG(click) rate FROM features GROUP BY 1"},
    ]
    responses = [
        action("sql", query="SELECT after_nonclick,COUNT(*) n,AVG(click) r FROM features GROUP BY 1"),
        action("sql", query="SELECT topic_changed,COUNT(*) n,AVG(click) r FROM features GROUP BY 1"),
        action("sql", query="SELECT after_long_gap,COUNT(*) n,AVG(click) r FROM features GROUP BY 1"),
        {"concepts": claims},
        action("test", concept_id="H001", query=claims[0]["suggested_sql"]),
        {"decision": "revise", "reason": "time controls needed",
         "next_sql": "", "observable_valid": True,
         "rival_discriminated": False, "logging_artifact_excluded": False},
        action("revise", concept_id="H001", concept="conditional state shift",
               mechanism="feedback persistence under long gaps",
               rival="user mix", falsifier="state effect disappears by user",
               measurement="click interaction"),
        action("test", concept_id="H001", query="SELECT after_nonclick,COUNT(*) n,AVG(click) rate FROM features GROUP BY 1"),
        {"decision": "ready", "reason": "pilot checks testable",
         "next_sql": "", "observable_valid": True,
         "rival_discriminated": True, "logging_artifact_excluded": True},
        action("stop"),
    ]
    seen_stages = []
    class FakeModel:
        def ask(self, prompt, schema, context):
            seen_stages.append(context["stage"])
            return responses.pop(0)
    monkeypatch.setattr(o, "create_model", lambda *a: FakeModel())
    result = o.run_open_discovery(str(source), str(tmp_path / "run"), {},
                                  max_rounds=6, development_path=str(db))
    assert result["status"] == "development_inconclusive"
    assert len(result["concepts"]) == 2
    assert result["concepts"][0]["version"] == 2
    assert len(result["concepts"][0]["tests"]) == 2
    assert "concept-formation" in seen_stages
    assert "discovery-critic" in seen_stages
    assert (tmp_path / "run" / "concept_formation.json").exists()
    assert not result["holdout_consulted"]

def test_reflection_revises_failed_scientific_mechanism(tmp_path, monkeypatch):
    source = _raw(tmp_path)
    dev = tmp_path / "dev.duckdb"
    o.development_db(str(source), str(dev))
    claim = dict(concept_id="H001", concept="raw tag effect",
                 mechanism="tag switch changes clicks", rival="user propensity",
                 falsifier="group gap disappears after control",
                 measurement="tag transition click")
    q1 = "SELECT topic_changed,COUNT(*) n,AVG(click) r FROM features GROUP BY 1"
    q2 = "SELECT after_nonclick,COUNT(*) n,AVG(click) r FROM features GROUP BY 1"
    q3 = "SELECT after_long_gap,COUNT(*) n,AVG(click) r FROM features GROUP BY 1"
    q4 = "SELECT event_day,COUNT(*) n,AVG(click) r FROM features GROUP BY 1"
    q5 = "SELECT activity,after_nonclick,COUNT(*) n,AVG(click) r FROM features GROUP BY 1,2"
    responses = [
        _act("sql", query="SELECT tags,COUNT(*) n FROM features GROUP BY 1"),
        _act("form_concept", **claim),
        _act("test", query=q1, **claim),
        {"decision": "test", "reason": "confounding unresolved", "next_sql": q2,
         "observable_valid": True, "rival_discriminated": False,
         "logging_artifact_excluded": False},
        {"decision": "revise", "reason": "first test warrants a better mechanism",
         "concept": "tag effect depends on context",
         "mechanism": "session boundary effect", "rival": "user selection",
         "falsifier": "control within users", "measurement": "session-conditioned click",
         "next_sql": q3},
        _act("test", query=q3, **claim),
        {"decision": "test", "reason": "lag confounding persists", "next_sql": q4,
         "observable_valid": True, "rival_discriminated": False,
         "logging_artifact_excluded": False},
        {"decision": "revise", "reason": "tag effect may be session dependent",
         "concept": "session-conditional tag change",
         "mechanism": "context boundary mediates tag impact",
         "rival": "user propensity and event logging",
         "falsifier": "same-user stratified effect vanishes",
         "measurement": "tag by session split with support",
         "next_sql": q5},
        _act("test", query=q5, **claim),
        {"decision": "ready", "reason": "test sufficiently discriminating for a pilot",
         "next_sql": "", "observable_valid": True, "rival_discriminated": True,
         "logging_artifact_excluded": True},
        _act("literature", **claim),
        {"search_queries": ["session conditional tag recommender", "feedback drift ranking"]},
        {"overlap": "partial", "unresolved_difference": "mechanism not identified",
         "strongest_prior_work_ids": [], "verdict": "insufficient_literature",
         "discriminating_next_test": "replay"},
        _act("freeze", **claim),
    ]
    class FakeModel:
        def ask(self, prompt, schema, context):
            return responses.pop(0)
    monkeypatch.setattr(o, "create_model", lambda *a: FakeModel())
    monkeypatch.setattr(o, "search_openalex", lambda qs: {
        "queries": qs, "papers": [], "errors": [{"query": "test", "error": "offline"}],
        "retrieval_complete": False})
    result = o.run_open_discovery(str(source), str(tmp_path / "run"), {},
                                  max_rounds=7, development_path=str(dev))
    assert result["status"] == "frozen_unconfirmed"
    assert result["concepts"][0]["version"] == 3
    assert result["concepts"][0]["revisions"][-1]["reason"].startswith("tag effect")
    assert result["concepts"][0]["literature"]["assessment"]["verdict"] == "insufficient_literature"
    journal = json.loads((tmp_path / "run" / "journal.json").read_text())
    assert any("reflection" in step for step in journal)


def test_exact_duplicate_query_rejected(tmp_path):
    source = _raw(tmp_path)
    dev = tmp_path / "dev.duckdb"
    o.development_db(str(source), str(dev))
    tools = o.DevelopmentTools(str(dev))
    try:
        tools.sql("SELECT COUNT(*) n FROM features")
        with pytest.raises(ValueError, match="already ran"):
            tools.sql("SELECT COUNT(*) n FROM features;")
    finally:
        tools.close()

def test_single_group_cannot_pass_scientific_critique(tmp_path, monkeypatch):
    source = _raw(tmp_path)
    dev = tmp_path / "dev.duckdb"
    o.development_db(str(source), str(dev))
    claim = dict(concept_id="H001", concept="one-group test",
                 mechanism="group effect", rival="composition",
                 falsifier="groups differ", measurement="group contrast")
    responses = [
        _act("sql", query="SELECT COUNT(*) n FROM features"),
        _act("form_concept", **claim),
        _act("test", query="SELECT AVG(click) rate FROM features", **claim),
        {"decision": "ready", "reason": "looks promising", "next_sql": "",
         "observable_valid": True, "rival_discriminated": True,
         "logging_artifact_excluded": True},
        _act("freeze", **claim),
    ]
    class FakeModel:
        def ask(self, prompt, schema, context):
            return responses.pop(0)
    monkeypatch.setattr(o, "create_model", lambda *args: FakeModel())
    result = o.run_open_discovery(str(source), str(tmp_path / "run"), {},
                                  max_rounds=4, development_path=str(dev))
    assert result["status"] == "development_inconclusive"
    journal = json.loads((tmp_path / "run" / "journal.json").read_text())
    assert journal[2]["contrast_audit"]["has_two_result_groups"] is False
    assert journal[2]["critique"]["observable_valid"] is False

def test_literature_retrieval_excludes_irrelevant_click_homonyms(monkeypatch):
    import io
    payload = {"results": [
        {"id": "https://openalex.org/Wd", "doi": None,
         "display_name": "Click acoustics of tagged dolphins",
         "publication_year": 2016,
         "abstract_inverted_index": {"dolphins": [0], "acoustics": [1]}},
        {"id": "https://openalex.org/Wr", "doi": "https://doi.org/10.1/xyz",
         "display_name": "Session-Based Recommender Systems and Feedback",
         "publication_year": 2022,
         "abstract_inverted_index": {"sequential": [0], "recommendation": [1]}},
    ]}
    def fake_urlopen(request, timeout):
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(o.urllib.request, "urlopen", fake_urlopen)
    result = o.search_openalex(["tag transitions affect clicks"], per_query=2)
    assert len(result["papers"]) == 1
    assert result["papers"][0]["id"] == "https://openalex.org/Wr"
    assert result["excluded_irrelevant"][0]["id"] == "https://openalex.org/Wd"
    assert "recommender systems" in result["executed_queries"][0]
