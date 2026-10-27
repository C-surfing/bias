"""Real Workbench contract: native observed groups need no invented derived feature."""
import duckdb

from rsi_researcher.research.discovery.experiment_contract import (
    compile_experiment_intent, experiment_intent_schema)
from rsi_researcher.research.discovery.open_research import DevelopmentTools
from rsi_researcher.research.discovery.workbench import ExperimentalWorkbench


def test_native_measured_comparison_compiles_to_event_rows(tmp_path):
    path = str(tmp_path / "native.duckdb")
    db = duckdb.connect(path)
    db.execute("CREATE TABLE events AS SELECT i//20 uid, i vid, "
               "20220423 event_day, i%2 is_random, i%3 click "
               "FROM range(400) t(i)")
    db.execute("CREATE TABLE features AS SELECT uid,vid,event_day,click FROM events")
    db.close()
    tools = DevelopmentTools(path)
    assert tools.feature_records == {}
    contrast = {
        "source_table": "events", "group_expression": "is_random",
        "control_value": "0", "treatment_value": "1",
        "outcome_expression": "click", "analysis": "paired_user",
    }
    intent = {
        "question": "Do exposure groups have distinct user-level click response?",
        "group_feature": "is_random", "control_value": "0",
        "treatment_value": "1", "outcome": "click", "population_filter": "",
        "analysis": "paired_user", "prediction": "H1 predicts positive gap",
        "falsifier": "H2 predicts a zero gap", "meaningful_margin": 0.01,
    }
    assert experiment_intent_schema(tools, contrast)["properties"][
        "group_feature"]["enum"] == ["is_random"]
    spec = compile_experiment_intent(intent, tools, contrast)
    assert spec["source_query"] == ""
    assert spec["unit_expression"] == "uid"
    result = ExperimentalWorkbench(tools).run(spec)
    assert result["result"]["n_paired_units"] == 20
    tools.close()
