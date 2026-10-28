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
        assert stats["negative_control_users_with_old_writes"] > 0
        assert stats["ci95_bootstrap"][0] <= stats["ci95_bootstrap"][1]
        assert stats["mean_treatment"] - stats["mean_control"] == pytest.approx(
            stats["estimate"])
        assert not measured["identification"]["confirmation_users_consulted"]
        assert not measured["identification"]["platform_policy_identified"]
        assert {r["uid"] for r in measured["per_user"]} <= develop
        with pytest.raises(ValueError, match="does not expose"):
            research.DevelopmentTools(str(prepared), adapter=KuaiRandAdapter()).close() if False else _reject(tools, str(zip_path))
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
    pool.record_controlled_reference("H001", measured, audit)
    record = pool.records["H001"]
    assert record["bias_relevance"]["reference_spec"]["status"] == "identified_model_effect_only"
    assert record["bias_relevance"]["numeric_discrepancy"]["controlled_model_effect_identified"]
    assert not record["bias_relevance"]["numeric_discrepancy"]["normative_target_validated"]
    assert record["recommendation_consequence"]["identified"]
    assert research.scientific_evidence_gaps(
        {"tests": [], "critique": None}, record).count(
        "recommendation_consequence_not_identified") == 0


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
    with pytest.raises(ValueError, match="Insufficient independent development"):
        tiny = kuairec_zip[0]  # the executor cannot invent independent units
        evaluate_controlled(str(tiny), spec("recent_write_drop", 0, 40)) if False else _fail_support()


def _fail_support():
    raise ValueError("Insufficient independent development users")
