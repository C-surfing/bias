"""Dataset-independent discovery and reference contracts, with real DuckDB paths."""
import json
import zipfile

import duckdb
import pytest

from rsi_researcher.research.discovery.datasets import (
    DatasetAdapter, KuaiRandAdapter, KuaiRecAdapter, get_dataset_adapter,
)
from rsi_researcher.research.discovery.references import (
    ReferenceSpec, EstimandSpec, rival_prediction_audit,
)
from rsi_researcher.research.discovery.opportunities import ResearchOpportunityPool
from rsi_researcher.research.discovery.real_intervention import (
    bucket, reference_panel,
)
from rsi_researcher.research.discovery import open_research as research


def test_reference_capability_and_estimand_alignment():
    choice = {
        "reference_kind": "randomized_exposure",
        "target_reference": "eligible random items",
        "proposed_distortion": "allocation share",
        "contrast_filter": "event_day=1",
        "contrast_source": "events", "contrast_group": "tags",
        "control_value": "a", "treatment_value": "b",
        "contrast_outcome": "click", "analysis": "paired_user",
    }
    with pytest.raises(ValueError, match="does not support"):
        ReferenceSpec.from_choice(choice, KuaiRecAdapter.capabilities)
    spec = ReferenceSpec.from_choice(choice, KuaiRandAdapter.capabilities)
    assert spec.status == "proposed_unidentified"
    assert spec.source_scope == "restricted_random_insertion_not_normative"
    estimand = EstimandSpec.from_choice(choice)
    executed = {**estimand.as_record(), "population_filter": "event_day=2"}
    audit = estimand.compare_executed(executed)
    assert not audit["same_estimand"] and audit["same_core_contrast"]
    assert audit["population_shifted"]
    executed["outcome_expression"] = "long_view"
    assert not estimand.compare_executed(executed)["same_core_contrast"]


def test_opportunity_preserves_reference_identification_assumptions():
    pool = ResearchOpportunityPool()
    pool.register("H001", {
        "concept": "candidate bias", "mechanism": "ranking update",
        "rival": "preferences", "falsifier": "no difference",
        "measurement": "allocation share", "suggested_sql": "SELECT 1",
        "reference_identification": "only within matched random item pool",
    }, [])
    choice = {
        "reference_kind": "randomized_exposure",
        "target_reference": "random insertion response",
        "proposed_distortion": "allocation shift",
        "contrast_source": "events", "contrast_group": "tags",
        "contrast_outcome": "click", "control_value": "a",
        "treatment_value": "b", "analysis": "paired_user",
        "contrast_filter": "",
        "mechanism_direction": "increase", "rival_direction": "decrease",
    }
    pool.record_selection_design(
        "H001", choice, {"valid_support": True},
        KuaiRandAdapter.capabilities)
    ref = pool.records["H001"]["bias_relevance"]["reference_spec"]
    assert ref["identification_assumptions"] == (
        "only within matched random item pool")
    assert ref["status"] == "proposed_unidentified"


def test_numeric_prediction_requires_positive_evidence_not_pvalue_alone():
    confident = {"ci95_bootstrap": [0.02, 0.08]}
    assert rival_prediction_audit(
        confident, "increase", "decrease", 0.01)["supports"] == "H1"
    crosses_zero = {"ci95_bootstrap": [-0.02, 0.03]}
    assert rival_prediction_audit(
        crosses_zero, "no_change", "increase", 0.01)["supports"] == "inconclusive"
    equivalent = {"ci95_bootstrap": [-0.005, 0.006]}
    assert rival_prediction_audit(
        equivalent, "no_change", "increase", 0.01)["supports"] == "H1"


def test_generic_adapter_refuses_unknown_holdout_copy(tmp_path):
    path = tmp_path / "development.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE events AS SELECT 1 uid, 2 vid, 1 click")
    con.execute("CREATE TABLE features AS SELECT 1 uid, 2 vid, 1 click")
    con.close()
    adapter = DatasetAdapter()
    with pytest.raises(ValueError, match="never copy"):
        adapter.prepare(str(path), str(tmp_path / "copied.duckdb"))
    assert adapter.prepare(str(path), str(path))["reused"]
    assert not adapter.install_reference_views(None, True)
    assert get_dataset_adapter("development_duckdb").dataset_id == "development_duckdb"


def _kuairec_fixture(tmp_path):
    path = tmp_path / "tiny-kuairec.zip"
    selected = [uid for uid in range(1, 100)
                if bucket(uid) == "development"][:3]
    confirmed = next(uid for uid in range(1, 100) if bucket(uid) == "confirmation")
    categories = "video_id,feat\n" + "\n".join(
        f'{vid},"[1, 2]"' for vid in range(1, 21)) + "\n"
    lines = ["user_id,video_id,watch_ratio,timestamp"]
    for uid in [*selected, confirmed]:
        for vid in range(1, 21):
            lines.append(f"{uid},{vid},{1.2 if vid%2 else 0.2},1660000000")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("KuaiRec 2.0/data/item_categories.csv", categories)
        z.writestr("KuaiRec 2.0/data/small_matrix.csv", "\n".join(lines) + "\n")
    return path, selected, confirmed


def test_kuairec_preparation_keeps_confirmation_and_item_panel_private(tmp_path):
    src, users, excluded = _kuairec_fixture(tmp_path)
    dest = tmp_path / "dev.duckdb"
    result = KuaiRecAdapter().prepare(str(src), str(dest))
    assert result["rows"]["events"] > 0
    db = duckdb.connect(str(dest), read_only=True)
    found = {row[0] for row in db.execute("SELECT DISTINCT uid FROM events").fetchall()}
    assert found and found <= set(users)
    assert excluded not in found
    for uid, vid in db.execute("SELECT uid,vid FROM events").fetchall():
        assert not reference_panel(uid, vid)
    assert {row[0] for row in db.execute("SHOW TABLES").fetchall()} == {
        "events", "features"}
    db.close()
    tools = research.DevelopmentTools(str(dest), adapter=KuaiRecAdapter())
    try:
        with pytest.raises(ValueError, match="No future or current outcome"):
            tools.define_feature("leaked_label_group",
                                 "CASE WHEN relevance=1 THEN 1 ELSE 0 END",
                                 source_table="events")
        with pytest.raises(ValueError, match="No future or current outcome"):
            tools.define_feature("leaked_watch_group",
                                 "CASE WHEN watch_ratio>=1 THEN 1 ELSE 0 END",
                                 source_table="features")
    finally:
        tools.close()


def test_same_discovery_loop_on_kuairec_without_random_exposure(tmp_path, monkeypatch):
    src, _, _ = _kuairec_fixture(tmp_path)
    actions = iter([
        {"action": "sql", "query": "SELECT uid,AVG(watch_ratio) response FROM events GROUP BY uid"},
        {"action": "sql", "query": "SELECT vid,AVG(relevance) response FROM events GROUP BY vid"},
        {"action": "stop", "query": ""},
    ])
    class Model:
        def ask(self, prompt, schema, context):
            assert context["stage"] == "open-discovery-scientist"
            item = next(actions)
            return {
                "action": item["action"], "reason": "development evidence only",
                "query": item["query"], "feature_name": "", "expression": "",
                "source_table": "events", "concept_id": "", "concept": "",
                "mechanism": "", "rival": "", "falsifier": "", "measurement": "",
            }
    monkeypatch.setattr(research, "create_model", lambda *args: Model())
    output = tmp_path / "result"
    result = research.run_open_discovery(
        str(src), str(output),
        {"discovery": {"dataset": "kuairec"}},
        max_rounds=3, min_queries=1)
    assert len(result["development_queries"]) == 2
    assert result["holdout_consulted"] is False
    manifest = json.loads((output / "development_manifest.json").read_text())
    assert manifest["dataset"]["dataset_id"] == "kuairec"
    assert manifest["dataset"]["capabilities"]["dense_relevance"] is True
    assert manifest["dataset"]["capabilities"]["randomized_exposure_probe"] is False


def test_random_insertion_reference_is_executed_and_estimand_bound(tmp_path):
    """Observed policy interaction is testable; it is NOT normative platform Bias."""
    from rsi_researcher.research.discovery.references import audit_reference_probe
    from rsi_researcher.research.discovery.workbench import randomized_reference_probe
    path = str(tmp_path / "random-mode.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE events(uid INTEGER, vid INTEGER, event_day INTEGER, "
                "is_random INTEGER, item_group VARCHAR, click INTEGER)")
    rows = []
    for uid in range(1, 25):
        for mode in (0, 1):
            for group in ("0", "1"):
                rows.append((uid, 100+int(group), 20220423, mode, group,
                             int(mode == 0 and group == "1")))
        # This video was NEVER shown by the standard recommender; it
        # must not enter a matched-policy inference or dilute the effect.
        rows.append((uid, 999, 20220423, 1, "1", 0))
    con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", rows)
    con.execute("CREATE TABLE features AS SELECT * FROM events")
    con.close()
    tools = research.DevelopmentTools(
        path, enable_matched_pool=True, adapter=KuaiRandAdapter())
    try:
        choice = {
            "reference_kind": "randomized_exposure",
            "contrast_source": "events", "contrast_group": "item_group",
            "contrast_outcome": "click", "contrast_filter": "",
            "control_value": "0", "treatment_value": "1",
            "mechanism_direction": "increase", "rival_direction": "decrease",
            "meaningful_margin": .01}
        probe = randomized_reference_probe(tools, choice)
        audit = audit_reference_probe(choice, probe)
        assert probe["matched_item_day_support"]
        assert probe["matched_item_day_strata"] == 2
        assert probe["n_user_matched_four_cell_units"] == 24
        assert probe["user_matched_interaction_supported"]
        assert probe["user_matched_policy_response_interaction"] == pytest.approx(1.)
        assert probe["user_matched_interaction_ci95"][0] > 0
        assert audit["matches_selected_observable"]
        assert audit["prediction_audit"]["supports"] == "H1"
        assert not audit["reference_identifies_production_bias"]
        pool = ResearchOpportunityPool()
        pool.register("H001", {
            "concept": "item opportunity discrepancy",
            "mechanism": "policy changes allocation",
            "rival": "content preferences", "falsifier": "no interaction",
            "measurement": "policy by item-group response",
            "suggested_sql": "SELECT 1"}, [])
        pool.record_selection_design("H001", {
            **choice, "target_reference": "restricted insertion panel",
            "proposed_distortion": "mode response interaction",
            "analysis": "paired_user"}, {"valid_support": True},
            KuaiRandAdapter.capabilities)
        pool.record_reference_diagnostic("H001", probe, audit)
        record = pool.records["H001"]["bias_relevance"]
        assert record["reference_spec"]["status"] == "measured_restricted_reference_only"
        assert not record["numeric_discrepancy"]["normative_target_validated"]
        wrong = audit_reference_probe({**choice, "contrast_outcome": "hate"}, probe)
        assert not wrong["matches_selected_observable"]
        assert wrong["prediction_audit"]["supports"] == "inconclusive"
        no_pool = audit_reference_probe(choice, {
            **probe, "matched_item_day_support": False})
        assert not no_pool["matches_selected_observable"]
        assert no_pool["prediction_audit"]["supports"] == "inconclusive"
        # The Agent may derive a novel item-side grouping; it must remain
        # executable against the SAME matched pool.
        tools.define_feature(
            "derived_item_arm",
            "CASE WHEN item_group='1' THEN '1' ELSE '0' END",
            source_table="events")
        derived = randomized_reference_probe(
            tools, {**choice, "contrast_group": "derived_item_arm"})
        assert derived["matched_item_day_strata"] == 2
        assert derived["standard_minus_random_response_contrast"] == pytest.approx(1.)
    finally:
        tools.close()


def test_random_reference_rejects_disjoint_item_day_support(tmp_path):
    from rsi_researcher.research.discovery.workbench import randomized_reference_probe
    path = str(tmp_path / "no-overlap.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE events(uid INT, vid INT, event_day INT, "
                "is_random INT, item_group VARCHAR, click INT)")
    con.execute("INSERT INTO events VALUES "
                "(1, 101, 20220423, 0, '0', 1), "
                "(2, 202, 20220423, 1, '1', 0)")
    con.execute("CREATE TABLE features AS SELECT * FROM events")
    con.close()
    tools = research.DevelopmentTools(
        path, enable_matched_pool=True, adapter=KuaiRandAdapter())
    try:
        choice = {"reference_kind": "randomized_exposure",
                  "contrast_source": "events", "contrast_group": "item_group",
                  "contrast_outcome": "click", "contrast_filter": "",
                  "control_value": "0", "treatment_value": "1"}
        with pytest.raises(ValueError, match="No shared item/day"):
            randomized_reference_probe(tools, choice)
    finally:
        tools.close()
