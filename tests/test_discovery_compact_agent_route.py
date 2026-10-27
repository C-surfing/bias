"""An actual Agent action chain must reach typed Workbench statistics."""
import duckdb
import json

from rsi_researcher.research.discovery import open_research as research


def action(name, **kwargs):
    fields = {"action": name, "reason": "test executable rival-discriminating idea",
              "query": "", "feature_name": "", "expression": "",
              "source_table": "events", "concept_id": "",
              "concept": "", "mechanism": "", "rival": "",
              "falsifier": "", "measurement": ""}
    fields.update(kwargs)
    return fields


def test_qwen_style_intent_executes_through_scientist_designer_critic(tmp_path, monkeypatch):
    db = str(tmp_path / "dev.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE TABLE events(uid INT, vid INT, event_day INT, "
                "is_random INT, tags VARCHAR, click INT, previous_click INT)")
    data = [(u, u * 100 + j, 20220423, j % 2,
             "A" if j % 4 < 2 else "B", 1 if j % 4 < 2 else 0, j % 2)
            for u in range(1, 21) for j in range(20)]
    con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?)", data)
    con.execute("CREATE TABLE features AS SELECT uid,click,"
                " 'active'::VARCHAR activity,0 after_long_gap FROM events")
    con.close()
    monkeypatch.setattr(research, "development_db",
                        lambda src, dst: {"path": db, "reused": True})

    scientific = {"concept": "conditional item allocation mismatch",
                  "mechanism": "recommender might underallocate an item group",
                  "rival": "preference and supply differences explain the gap",
                  "falsifier": "gap disappears after conditioning",
                  "measurement": "item-side exposure versus response",
                  "suggested_sql": "SELECT tags,COUNT(*) FROM events GROUP BY 1",
                  "target_reference": "unmeasured",
                  "reference_identification": "no justified normative reference yet",
                  "distortion_quantity": "allocation share difference",
                  "distortion_estimate": "unmeasured",
                  "prediction_if_mechanism": "within-user gap persists",
                  "prediction_if_rival": "conditional gap disappears"}
    compact = {
        "question": "Does actual item content group affect user response?",
        "group_feature": "item_group", "control_value": "0",
        "treatment_value": "1", "outcome": "click",
        "population_filter": "", "analysis": "paired_user",
        "prediction": "H1 says the conditional gap persists",
        "falsifier": "H2 says the conditional gap disappears",
        "meaningful_margin": 0.01}
    chooser = {
        "selected_concept_id": "H001",
        "reason": "conditioned observation separates rival predictions",
        "discriminating_prediction": "within-user conditional item response",
        "prediction_if_mechanism": "gap increases",
        "prediction_if_rival": "gap remains unchanged",
        "why_not_other": "other option has less support",
        "next_action": "experiment", "next_sql": "", "experimental_cost": "low",
        "contrast_source": "events", "contrast_group": "item_group",
        "contrast_outcome": "click", "contrast_filter": "",
        "control_value": "0", "treatment_value": "1", "analysis": "paired_user",
        "mechanism_direction": "increase", "rival_direction": "no_change",
        "reference_kind": "none", "target_reference": "unmeasured",
        "proposed_distortion": "conditional underallocation candidate",
        "mechanism_entry_point": "ranking",
        "corrective_intervention": "not tested"}

    stages = []
    class FakeModel:
        def __init__(self):
            self.scientist_i = 0
        def ask(self, prompt, schema, context):
            role = context["stage"]
            stages.append(role)
            if role == "open-discovery-scientist":
                decisions = [
                    action("define_feature", feature_name="item_group",
                           expression="CASE WHEN tags='A' THEN 1 ELSE 0 END"),
                    action("sql", query="SELECT tags,COUNT(*) n FROM events GROUP BY 1"),
                    action("sql", query="SELECT is_random,COUNT(*) n FROM events GROUP BY 1"),
                    action("experiment", concept_id="H001"),
                    action("stop")]
                value = decisions[self.scientist_i]
                self.scientist_i += 1
                return value
            if role == "concept-formation":
                return {"concepts": [scientific, {**scientific,
                    "concept": "competing effect under changed user context"}]}
            if role == "research-opportunity-selector":
                return chooser
            if role == "research-designer":
                assert "group_feature" in schema["properties"]
                return compact
            if role == "discovery-critic":
                return {"decision": "ready", "reason": "descriptive only",
                        "next_sql": "", "observable_valid": True,
                        "rival_discriminated": False,
                        "logging_artifact_excluded": False}
            raise AssertionError(role)

    fake=FakeModel()
    monkeypatch.setattr(research,"create_model",lambda *args: fake)
    output=tmp_path/"run"
    result=research.run_open_discovery("unused",str(output),
        {"llm":{"provider":"vllm"}},max_rounds=5,development_path=db)
    journal=json.loads((output/"journal.json").read_text())
    experiments=[x for x in journal if "experiment" in x]
    assert len(experiments)==1
    assert experiments[0]["experiment"]["result"]["n_paired_units"]==20
    assert experiments[0]["experiment"]["identifies_causal_bias"] is False
    assert {"concept-formation","research-opportunity-selector",
            "research-designer","discovery-critic"} <= set(stages)
    assert result["status"]=="development_inconclusive"
    assert result["research_funnel"]["candidates_with_executed_workbench"]==1
