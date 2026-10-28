"""End-to-end controlled references on a KuaiRec-format development fixture."""
import json
import zipfile

import duckdb
import pytest

from rsi_researcher.research.discovery import open_research as research
from rsi_researcher.research.discovery.controlled_reference import (
    evaluate_controlled, validate_controlled_spec,
)
from rsi_researcher.research.discovery.datasets import KuaiRecAdapter, KuaiRandAdapter
from rsi_researcher.research.discovery.opportunities import ResearchOpportunityPool
from rsi_researcher.research.discovery.real_intervention import bucket, reference_panel
from rsi_researcher.research.discovery.workbench import ExperimentalWorkbench
from rsi_researcher.research.discovery.references import rival_prediction_audit


@pytest.fixture
def kuairec_zip(tmp_path):
    path = tmp_path / "controlled-kuairec.zip"
    develop = [u for u in range(1, 500) if bucket(u) == "development"][:18]
    train = [u for u in range(1, 500) if bucket(u) == "train"][:12]
    confirmation = next(u for u in range(1, 500) if bucket(u) == "confirmation")
    assert len(develop) == 18
    cats = "video_id,feat\n" + "\n".join(
        f'{vid},"[{1 if vid%2 else 2}]"' for vid in range(1, 121)) + "\n"
    lines = ["user_id,video_id,watch_ratio,timestamp"]
    for uid in [*train, *develop, confirmation]:
        for vid in range(1, 121):
            reward = 1.2 if ((uid + vid) % 7 < 3 or (uid%3==0 and vid%2==1)) else 0.2
            lines.append(f"{uid},{vid},{reward},1660000000")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("KuaiRec 2.0/data/item_categories.csv", cats)
        z.writestr("KuaiRec 2.0/data/small_matrix.csv", "\n".join(lines)+"\n")
    return path, set(develop), confirmation


def spec(parameter="profile_strength", a=0, b=1.2):
    return {
        "question": "Do model history signals affect held-out ranking quality?",
        "parameter": parameter, "control_value": a, "treatment_value": b,
        "outcome": "ndcg@10", "prediction_if_mechanism": "increase",
        "prediction_if_rival": "decrease", "meaningful_margin": 0.001,
        "reference_justification": "Same users and hidden fixed relevance panel",
    }


def test_controlled_recommender_has_an_actual_paired_model_reference(kuairec_zip, tmp_path):
    zip_path, develop, confirmation = kuairec_zip
    prepared = tmp_path / "dev.duckdb"
    KuaiRecAdapter().prepare(str(zip_path), str(prepared))
    db = duckdb.connect(str(prepared), read_only=True)
    ids = {r[0] for r in db.execute("SELECT DISTINCT uid FROM events").fetchall()}
    assert ids == develop and confirmation not in ids
    for uid, vid in db.execute("SELECT uid,vid FROM events").fetchall():
        assert not reference_panel(uid, vid)
    db.close()
    tools = research.DevelopmentTools(str(prepared), adapter=KuaiRecAdapter())
    try:
        measured = ExperimentalWorkbench(tools).run_controlled_reference(
            str(zip_path), spec())
        stats = measured["result"]
        assert stats["n_paired_units"] >= 12
        assert stats["negative_control_passed"]
        assert 0 < stats["negative_control_users_with_old_writes"] <= stats["n_paired_units"]
        assert measured["estimand"]["outcome"] == "ndcg@10"
        assert measured["estimand"]["statistical_unit"] == "uid"
        assert measured["estimand"]["phase"] == "development"
        assert stats["ci95_bootstrap"][0] <= stats["ci95_bootstrap"][1]
        assert stats["mean_treatment"] - stats["mean_control"] == pytest.approx(
            stats["estimate"])
        assert not measured["identification"]["confirmation_users_consulted"]
        assert not measured["identification"]["platform_policy_identified"]
        assert {r["uid"] for r in measured["per_user"]} <= develop
        with pytest.raises(ValueError, match="does not expose"):
            _reject(tools, str(zip_path))
    finally:
        tools.close()


def _reject(tools, zip_path):
    adapter = tools.adapter
    tools.adapter = KuaiRandAdapter()
    try:
        return ExperimentalWorkbench(tools).run_controlled_reference(zip_path, spec())
    finally:
        tools.adapter = adapter


def test_controlled_state_can_identify_only_the_scoped_model_effect(kuairec_zip):
    zip_path, _, _ = kuairec_zip
    measured = evaluate_controlled(str(zip_path), spec("recent_write_drop", 0, 8))
    pool = ResearchOpportunityPool()
    pool.register("H001", {
        "concept": "unknown conditional distortion", "mechanism": "feedback pathway",
        "rival": "user preferences", "falsifier": "replay has no effect",
        "measurement": "heldout ranking quality", "suggested_sql": "SELECT 1",
    }, [])
    choice = {
        "reference_kind": "controlled_model",
        "target_reference": "fixed independently labelled ranking panel",
        "proposed_distortion": "within-model NDCG contrast",
        "contrast_source": "events", "contrast_group": "tags",
        "contrast_outcome": "relevance", "contrast_filter": "",
        "control_value": "1", "treatment_value": "2", "analysis": "paired_user",
        "mechanism_direction": "increase", "rival_direction": "decrease",
    }
    pool.record_selection_design("H001", choice, {"valid_support": True},
                                 KuaiRecAdapter.capabilities)
    audit = rival_prediction_audit(
        measured["result"], "increase", "decrease", 0.001)
    from copy import deepcopy
    tampered = deepcopy(measured)
    tampered["estimand"]["outcome"] = "recall@10"
    with pytest.raises(ValueError, match="estimand differs"):
        pool.record_controlled_reference("H001", tampered, audit)
    pool.record_controlled_reference("H001", measured, audit)
    record = pool.records["H001"]
    assert record["bias_relevance"]["reference_spec"]["status"] == "identified_model_effect_only"
    assert record["bias_relevance"]["numeric_discrepancy"]["controlled_model_effect_identified"]
    assert not record["bias_relevance"]["numeric_discrepancy"]["normative_target_validated"]
    assert record["recommendation_consequence"]["identified"]
    assert research.scientific_evidence_gaps(
        {"tests": [], "critique": None}, record).count(
        "recommendation_consequence_not_identified") == 0


def test_placebo_is_unavailable_not_passed_when_no_old_history(kuairec_zip):
    zip_path, _, _ = kuairec_zip
    measured = evaluate_controlled(str(zip_path),
                                   spec("history_window", 105, 115))
    stats = measured["result"]
    assert stats["n_paired_units"] >= 12
    assert not stats["negative_control_tested"]
    assert stats["negative_control_passed"] is None
    assert stats["negative_control_users_with_old_writes"] == 0


def test_invalid_or_non_discriminating_model_design_is_rejected(kuairec_zip):
    zip_path, _, _ = kuairec_zip
    for changed in [
        {**spec(), "treatment_value": 0},
        {**spec(), "control_value": float("nan")},
        {**spec(), "parameter": "history_window", "treatment_value": 10.5},
        {**spec(), "prediction_if_rival": "increase"},
    ]:
        with pytest.raises(ValueError):
            validate_controlled_spec(changed)
    # A real evaluator must refuse under-supported cohorts; it cannot
    # turn many events from one user into independent research samples.


def test_agent_controls_the_experiment_and_evidence_enters_scientific_loop(
        kuairec_zip, tmp_path, monkeypatch):
    zip_path, development_users, _ = kuairec_zip

    def action(name, concept_id="", query=""):
        return {
            "action": name, "reason": "investigate measured recommender behavior",
            "query": query, "feature_name": "", "expression": "",
            "source_table": "events", "concept_id": concept_id,
            "concept": "", "mechanism": "", "rival": "",
            "falsifier": "", "measurement": "",
        }

    initial = {
        "concept": "unexplained model-response ranking discrepancy",
        "mechanism": "user-state weighting changes ranking utility",
        "rival": "item priors alone explain observed quality",
        "falsifier": "altering the user-state coefficient cannot change ranked quality",
        "measurement": "paired fixed-candidate ranking utility",
        "suggested_sql": "SELECT tags,AVG(relevance) FROM events GROUP BY tags",
        "target_reference": "fixed relevance panel in constructed recommender",
        "reference_identification": "same users, candidates and relevance labels",
        "distortion_quantity": "difference in heldout NDCG",
        "distortion_estimate": "unmeasured",
        "prediction_if_mechanism": "ranking quality changes after model state intervention",
        "prediction_if_rival": "ranking quality is unaffected",
        "affected_population": "development users with enough history",
        "mechanism_entry_point": "ranking score state coefficient",
        "reference_status": "unmeasured",
        "next_information_needed": "paired controlled intervention",
        "discriminating_experiment": "compare fixed panel scores with changed profile weight",
    }
    choice = {
        "selected_concept_id": "H001", "reason": "test owned model state",
        "discriminating_prediction": "relevance gap for matched candidate groups",
        "prediction_if_mechanism": "different response groups",
        "prediction_if_rival": "same response groups",
        "why_not_other": "more direct model-control experiment",
        "next_action": "experiment", "next_sql": "",
        "experimental_cost": "paired replay",
        "contrast_source": "events", "contrast_group": "tags",
        "contrast_outcome": "relevance", "contrast_filter": "",
        "control_value": "1", "treatment_value": "2",
        "analysis": "paired_user",
        "mechanism_direction": "increase", "rival_direction": "no_change",
        "reference_kind": "controlled_model",
        "target_reference": "fixed relevance panel",
        "proposed_distortion": "controlled utility change",
        "mechanism_entry_point": "score state",
        "corrective_intervention": "not assumed",
    }

    stages = []
    class FakeModel:
        def __init__(self):
            self.actions = iter([
                action("sql", query="SELECT tags,COUNT(*) n FROM events GROUP BY tags"),
                action("sql", query="SELECT AVG(relevance) p FROM events"),
                action("sql", query="SELECT uid,COUNT(*) n FROM events GROUP BY uid"),
                action("controlled_experiment", concept_id="H001"),
                action("stop"),
            ])

        def ask(self, prompt, schema, context):
            stage = context["stage"]
            stages.append(stage)
            if stage == "open-discovery-scientist":
                answer = next(self.actions)
                assert answer["action"] in schema["properties"]["action"]["enum"]
                return answer
            if stage == "concept-formation":
                return {"concepts": [initial, {
                    **initial, "concept": "alternative latent mechanism",
                    "mechanism": "category recall response",
                    "measurement": "category-specific utility deviation"}]}
            if stage == "research-opportunity-selector":
                assert "controlled_model" in schema["properties"]["reference_kind"]["enum"]
                return choice
            if stage == "controlled-experiment-designer":
                return spec()
            if stage == "discovery-critic":
                return {
                    "decision": "revise", "reason": "only owned recommender identified",
                    "next_sql": "", "observable_valid": True,
                    "rival_discriminated": False,
                    "logging_artifact_excluded": False,
                }
            if stage == "discovery-reflection":
                return {
                    "decision": "revise", "reason": "test a narrower mechanism",
                    "concept": initial["concept"],
                    "mechanism": initial["mechanism"],
                    "rival": initial["rival"],
                    "falsifier": "replay should change results only for current contexts",
                    "measurement": "conditional within-model utility contrast",
                    "next_sql": "SELECT tags,AVG(relevance) FROM events GROUP BY tags",
                }
            raise AssertionError(stage)

    monkeypatch.setattr(research, "create_model", lambda *a: FakeModel())
    output = tmp_path / "agent-controlled"
    result = research.run_open_discovery(
        str(zip_path), str(output),
        {"llm": {"provider": "vllm"}, "discovery": {
            "dataset": "kuairec", "open_phenomena": False}},
        max_rounds=5, min_queries=1)
    journal = json.loads((output / "journal.json").read_text())
    actual = [x for x in journal if "controlled_experiment" in x]
    assert len(actual) == 1
    event = actual[0]
    assert "controlled-experiment-designer" in stages
    assert "discovery-critic" in stages
    assert "discovery-reflection" in stages
    assert event["controlled_experiment"]["result"]["n_paired_units"] >= 12
    assert event["numeric_prediction_audit"]["prediction_h1"] == "increase"
    assert event["reflection"]["decision"] == "revise"
    assert set(x["uid"] for x in event["controlled_experiment"]["per_user"]) <= development_users
    pool = result["research_opportunities"]["opportunities"]
    identified = next(x for x in pool if x["id"] == "H001")
    # The intervention identified the old model hypothesis, but Reflection
    # changed the target measurement. Previous evidence is retained by version;
    # it does not certify the new hypothesis without a new experiment.
    assert identified["controlled_reference_tests"][0]["tested_concept_version"] == 1
    assert identified["controlled_reference_tests"][0]["identification"][
        "within_model_paired_intervention"]
    assert identified["bias_relevance"]["reference_spec"]["status"] == "revised_requires_retest"
    assert identified["recommendation_consequence"]["identified"] is False
    assert "recommendation_consequence_not_identified" in identified["scientific_evidence_gaps"]
    assert result["research_funnel"]["executed_constructed_model_reference_tests"] == 1
    assert result["research_funnel"]["candidates_with_identified_constructed_model_effect"] == 0
    assert result["holdout_consulted"] is False
    assert result["frozen_id"] is None
    from rsi_researcher.research.discovery.review import assess
    review = assess(journal, {"concepts": result["concepts"]})
    assert review["controlled_model_workflow_executed"] is True
    assert review["scientific_quality"][
        "constructed_model_effect_identified_in_scope"] is True
    assert review["scientific_quality"][
        "independently_confirmed_recommender_bias"] is False
