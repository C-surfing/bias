"""Regression for PR15's mechanical click-history overlap and experiment design."""
import duckdb
import pytest

from rsi_researcher.research.discovery.active_experiments import (
    assess_candidates, choose_feasible, design_and_select,
)
from rsi_researcher.research.discovery.datasets import KuaiRandAdapter
from rsi_researcher.research.discovery.open_research import DevelopmentTools
from rsi_researcher.research.discovery.variable_lineage import (
    audit_construct_overlap, audit_observation_sql,
)


@pytest.fixture
def tools(tmp_path):
    dbpath = str(tmp_path / "development.duckdb")
    con = duckdb.connect(dbpath)
    con.execute("CREATE TABLE events(uid INTEGER,vid INTEGER,event_day INTEGER,"
                "moment_ms BIGINT,previous_click INTEGER,recent_click DOUBLE,"
                "click INTEGER,previous_hate INTEGER,is_random INTEGER)")
    con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)", [
        (uid, vid, 20220423, uid * 1000 + vid,
         vid % 2, (vid % 2 + 2) / 8, (vid + uid) % 2, 0, uid % 2)
        for uid in range(1, 31) for vid in range(1, 9)])
    con.execute("CREATE TABLE features AS SELECT * FROM events")
    con.close()
    dev = DevelopmentTools(
        dbpath, adapter=KuaiRandAdapter(), enable_matched_pool=True)
    yield dev
    dev.close()


def idea(sql, *, h1="increase", h2="decrease", decision=None, reference=None):
    plan = {
        "concept_id": "H001", "question": "Compare an H1/H2 hypothesis",
        "required_capability": "sql_observation",
        "study_unit": "user-event", "target_quantity": "response estimate",
        "prediction_h1": "higher outcome in group", "prediction_h2": "lower outcome in group",
        "direction_h1": h1, "direction_h2": h2, "sql": sql,
        "reference_assumption": "not yet verified",
        "mechanism_link": "ranking mechanism under investigation",
    }
    if decision is not None:
        plan["decision_target"] = decision
    if reference is not None:
        plan["reference_kind"] = reference
    return plan


def test_pr15_previous_click_recent_click_is_mechanical_overlap(tools):
    audit = audit_construct_overlap(tools, "previous_click", "recent_click")
    assert audit["status"] == "overlapping_constructs"
    assert audit["shared"][0]["source"] == "click"
    assert audit["shared"][0]["overlap_lags"] == [1, 1]
    assert audit_construct_overlap(tools, "previous_click", "click")[
        "status"] == "lineage_known_no_overlap"
    assert audit_construct_overlap(tools, "previous_hate", "click")[
        "status"] == "lineage_known_no_overlap"


def test_derived_history_group_inherits_lineage(tools):
    # No data execution required: use registered expression metadata.
    tools.feature_records["history_bucket"] = {
        "expression": "CASE WHEN previous_click = 1 THEN 1 ELSE 0 END",
    }
    audit = audit_construct_overlap(tools, "history_bucket", "recent_click")
    assert audit["status"] == "overlapping_constructs"
    assert audit_observation_sql(
        tools, "SELECT history_bucket, AVG(recent_click) FROM events GROUP BY 1"
    )["status"] == "overlapping_constructs"


def test_active_choice_filters_overlap_before_observing_results(tools):
    overlap = idea("SELECT previous_click,AVG(recent_click) "
                   "FROM events GROUP BY previous_click")
    supported = idea("SELECT is_random,vid,COUNT(*) n "
                     "FROM shared_item_pool_events GROUP BY 1,2",
                     decision="exposure_allocation", reference="random_insertions")
    behavior = idea("SELECT is_random,COUNT(*) n,AVG(click) "
                    "FROM events GROUP BY is_random",
                    decision="behavior_only", reference="none")
    checked = assess_candidates(
        tools, [{"id": "H001", "status": "investigating"}],
        [overlap, behavior, supported])
    assert [x["status"] for x in checked] == [
        "overlapping_constructs",
        "feasible_descriptive_observation",
        "feasible_descriptive_observation"]
    assert choose_feasible(checked)["candidate"] is supported
    assert checked[2]["decision_link_audit"][
        "measured_restricted_allocation"] is True
    assert tools.history == []  # preflight never reads outcomes


def test_one_repair_fixes_overlap_without_doubling_failed_actions(tools):
    bad = idea("SELECT previous_click,AVG(recent_click) "
               "FROM events GROUP BY previous_click")
    good = idea("SELECT previous_click,COUNT(*) n,AVG(click) "
                "FROM events GROUP BY previous_click")
    class Designer:
        calls = 0
        def act(self, context, tag, schema):
            self.calls += 1
            if self.calls == 2:
                assert context["rejected_design_audit"][0][
                    "status"] == "overlapping_constructs"
            return {"candidates": [bad] if self.calls == 1 else [good]}
    designer = Designer()
    chosen, checked, attempts = design_and_select(
        designer, {}, tools, [{"id": "H001", "status": "investigating"}], "test")
    assert designer.calls == 2
    assert len(attempts) == 2
    assert chosen["candidate"]["sql"] == good["sql"]
    assert tools.history == []


def test_unverified_lineage_is_not_claimed_safe(tools):
    audit = audit_construct_overlap(tools, "arbitrary_new_column", "click")
    assert audit["status"] == "lineage_unverified"


def test_false_decision_labels_cannot_upgrade_click_correlation(tools):
    falsely_claimed_allocation = idea(
        "SELECT previous_click,COUNT(*) n,AVG(click) FROM events GROUP BY 1",
        decision="exposure_allocation", reference="random_insertions")
    checked = assess_candidates(
        tools, [{"id": "H001", "status": "investigating"}],
        [falsely_claimed_allocation])
    assert checked[0]["status"] == "feasible_descriptive_observation"
    assert checked[0]["decision_reference_priority"] == 0
    assert checked[0]["scientific_scope"] == "exploratory_unidentified"
    assert checked[0]["decision_link_audit"]["measured_restricted_allocation"] is False
