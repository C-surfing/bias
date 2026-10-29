"""Regression tests for research selection -> actual experiment handoff."""
import unittest

from rsi_researcher.research.discovery.experiment_contract import compile_experiment_intent
from rsi_researcher.research.discovery.references import EstimandSpec
from rsi_researcher.research.discovery.research_handoff import (
    pending_primary_contract, previously_measured, primary_experiment_intent,
)


class ResearchHandoffTests(unittest.TestCase):
    def test_duplicate_sql_reuses_existing_evidence(self):
        old = {"sql": "SELECT count(*) FROM events", "rows": [{"count": 3}]}
        self.assertIs(
            previously_measured(" SELECT count(*) FROM events; ", [old]), old)
        self.assertIsNone(
            previously_measured("SELECT click FROM events", [old]))

    def test_pending_test_only_for_current_concept_version(self):
        concept = {"id": "H001", "status": "investigating", "version": 2,
                   "tests": [{"version": 1, "workbench": {"result": {}}}]}
        old = {"H001": {"selection_contract": {"concept_version": 1}}}
        self.assertIsNone(pending_primary_contract([concept], old))
        current = {"H001": {"selection_contract": {"concept_version": 2}}}
        self.assertEqual(pending_primary_contract([concept], current), "H001")
        concept["tests"].append({"version": 2, "workbench": {"result": {}}})
        self.assertIsNone(pending_primary_contract([concept], current))
        concept["status"] = "design_blocked"
        concept["tests"].clear()
        self.assertIsNone(pending_primary_contract([concept], current))

    def test_primary_intent_preserves_selected_estimand(self):
        estimand = {"source_table": "events", "unit_expression": "uid",
                    "population_filter": "previous_click = 0",
                    "group_expression": "item_group",
                    "control_value": "0", "treatment_value": "1",
                    "outcome_expression": "click", "analysis": "paired_user"}
        contract = {"estimand_spec": estimand,
                    "prediction_if_mechanism": "increase",
                    "prediction_if_rival": "decrease"}
        concept = {"claim": {"measurement": "item-group response difference"}}
        actual = primary_experiment_intent(contract, concept)
        self.assertEqual(actual["group_feature"], "item_group")
        self.assertEqual(actual["population_filter"], "previous_click = 0")
        self.assertEqual(actual["outcome"], "click")
        self.assertEqual(actual["prediction"], "H1: increase")
        self.assertEqual(actual["falsifier"], "H2: decrease")

        class FakeTools:
            feature_records = {}

            @staticmethod
            def schema():
                return {"events": [
                    {"column": "uid", "type": "VARCHAR"},
                    {"column": "item_group", "type": "INTEGER"},
                    {"column": "click", "type": "INTEGER"},
                ]}

        full_contract = {**estimand, **contract}
        compiled = compile_experiment_intent(
            actual, FakeTools(), full_contract)
        self.assertTrue(
            EstimandSpec(**estimand).compare_executed(compiled)["same_estimand"])


if __name__ == "__main__":
    unittest.main()
