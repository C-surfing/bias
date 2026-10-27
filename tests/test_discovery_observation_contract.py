"""Regression of real supported item-side intent, no model or holdout."""
import duckdb
import pytest
from jsonschema import ValidationError

from rsi_researcher.research.discovery.open_research import DevelopmentTools
from rsi_researcher.research.discovery.observations import build_observation_cards
from rsi_researcher.research.discovery.experiment_contract import (
    compile_experiment_intent, experiment_intent_schema)
from rsi_researcher.research.discovery.workbench import ExperimentalWorkbench


def test_observation_cards_and_compact_intent_are_executable(tmp_path):
    db = str(tmp_path / "isolated_development.duckdb")
    conn = duckdb.connect(db)
    conn.execute("CREATE TABLE events(uid INTEGER, vid INTEGER, event_day INTEGER, "
                 "is_random INTEGER, tags VARCHAR, click INTEGER, "
                 "previous_click INTEGER)")
    rows = [(uid, uid * 100 + j, 20220423, j % 2,
             "tagA" if j % 4 < 2 else "tagB",
             1 if j % 4 < 2 else 0, j % 2)
            for uid in range(1, 21) for j in range(20)]
    conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?)", rows)
    conn.execute("CREATE TABLE features AS SELECT uid,click, "
                 "'active'::VARCHAR activity,0 AS after_long_gap FROM events")
    conn.close()

    tools = DevelopmentTools(db)
    try:
        cards = build_observation_cards(tools)
        assert {c["observation_id"] for c in cards} >= {"O001", "O003", "O004", "O005"}
        assert all(c["measurement_sql"] and c["evidence_level"]
                   == "descriptive_development_only" for c in cards)
        feature = tools.define_feature(
            "item_tag_binary", "CASE WHEN tags='tagA' THEN 1 ELSE 0 END", "events")
        assert len(feature["evidence"]["rows"]) == 2
        intent = {
            "question": "Does item group affect responses on the common dates?",
            "group_feature": "item_tag_binary",
            "control_value": "0",
            "treatment_value": "1",
            "outcome": "click",
            "population_filter": "event_day=20220423",
            "analysis": "paired_user",
            "prediction": "H1 predicts persistent difference",
            "falsifier": "H2 predicts no conditional difference",
            "meaningful_margin": 0.01,
        }
        schema = experiment_intent_schema(tools)
        assert len(schema["required"]) == 10
        with pytest.raises(ValidationError):
            compile_experiment_intent({**intent, "treatment_value": "invented"}, tools)
        with pytest.raises(ValidationError):
            compile_experiment_intent({**intent, "outcome": "tags"}, tools)
        spec = compile_experiment_intent(intent, tools)
        assert spec["source_query"] == "" and spec["unit_expression"] == "uid"
        measured = ExperimentalWorkbench(tools).run(spec, required_feature="item_tag_binary")
        assert measured["result"]["n_paired_units"] == 20
        assert measured["result"]["estimate"] == 1.0
        assert measured["identifies_causal_bias"] is False
    finally:
        tools.close()
