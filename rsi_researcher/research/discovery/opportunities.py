"""Evidence-linked research opportunities inside one open-discovery cycle.

The agent proposes mechanisms and competing explanations. This module only
preserves their provenance and trial outcomes; it does NOT enumerate bias
types, choose candidate variables, or silently rank by statistical significance.
"""
from __future__ import annotations


class ResearchOpportunityPool:
    def __init__(self):
        self.records: dict[str, dict] = {}
        self.focus: str | None = None

    def register(self, identifier: str, proposed: dict, observations: list[dict]):
        self.records[identifier] = {
            "id": identifier,
            "status": "proposed",
            "phenomenon": proposed["concept"],
            "mechanism": proposed["mechanism"],
            "rival": proposed["rival"],
            "falsifier": proposed["falsifier"],
            "measurement": proposed["measurement"],
            "observed_evidence": [{
                "sql": record.get("sql", ""),
                "rows": record.get("rows", [])[:8]} for record in observations[-4:]],
            "prediction_if_mechanism": proposed.get("prediction_if_mechanism", ""),
            "prediction_if_rival": proposed.get("prediction_if_rival", ""),
            "discriminating_experiment": proposed.get("discriminating_experiment", ""),
            "next_observation": proposed.get("next_observation", proposed["suggested_sql"]),
            "scientific_value": proposed.get("scientific_value", ""),
            "experiment_cost": proposed.get("experiment_cost", ""),
            "priority_rationale": proposed.get("priority_rationale", ""),
            "tests": [],
            "selection_history": [],
        }

    def choose(self, identifier: str, rationale: str, next_query: str = ""):
        if identifier not in self.records:
            raise ValueError("Select an existing research opportunity")
        self.focus = identifier
        opportunity = self.records[identifier]
        opportunity["status"] = "selected"
        opportunity["selection_history"].append({
            "rationale": rationale, "next_observation": next_query})
        return identifier

    def observation(self, identifier: str, sql: str, rows: list):
        if identifier not in self.records:
            return
        item = self.records[identifier]
        item["observed_evidence"].append({"sql": sql, "rows": rows[:8]})

    def experiment(self, identifier: str, initial: dict, counter: dict | None,
                   critic: dict):
        if identifier not in self.records:
            return
        item = self.records[identifier]
        item["status"] = "countertested" if counter else "tested"
        event = {
            "initial_estimate": initial["result"]["estimate"],
            "initial_ci": initial["result"]["ci95_bootstrap"],
            "initial_users": initial["result"]["n_paired_units"],
            "initial_specification": initial["specification"],
            "critic_decision": critic["decision"],
            "critic_reason": critic["reason"],
        }
        if counter is not None:
            event["counter_estimate"] = counter["result"]["estimate"]
            event["counter_ci"] = counter["result"]["ci95_bootstrap"]
            event["counter_users"] = counter["result"]["n_paired_units"]
            event["counter_specification"] = counter["specification"]
        item["tests"].append(event)

    def update_claim(self, identifier: str, concept: dict, status: str):
        if identifier not in self.records:
            return
        item = self.records[identifier]
        item["status"] = status
        item["mechanism"] = concept["mechanism"]
        item["rival"] = concept["rival"]
        item["falsifier"] = concept["falsifier"]
        item["measurement"] = concept["measurement"]
        if status in ("abandoned", "rejected_development"):
            if self.focus == identifier:
                self.focus = None

    def snapshot(self) -> dict:
        return {"focus": self.focus, "opportunities": list(self.records.values())}

    def for_agent(self) -> dict:
        """Compact enough for Qwen without hiding the competing mechanisms."""
        return {"focus": self.focus, "opportunities": [
            {k: v for k, v in item.items() if k not in
             ("observed_evidence", "tests")} |
            {"observed_evidence": item["observed_evidence"][-2:],
             "tests": [{
                 "initial_estimate": t["initial_estimate"],
                 "initial_ci": t["initial_ci"],
                 "initial_users": t["initial_users"],
                 "counter_estimate": t.get("counter_estimate"),
                 "counter_ci": t.get("counter_ci"),
                 "counter_users": t.get("counter_users"),
                 "critic_reason": t["critic_reason"]
             } for t in item["tests"][-2:]]}
            for item in self.records.values()]}
