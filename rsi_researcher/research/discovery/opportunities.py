"""Evidence-linked research opportunities inside one open-discovery cycle.

The agent proposes mechanisms and competing explanations. This module only
preserves their provenance and trial outcomes; it does NOT enumerate bias
types, choose candidate variables, or silently rank by statistical significance.
"""
from __future__ import annotations

import math


class ResearchOpportunityPool:
    def __init__(self):
        self.records: dict[str, dict] = {}
        self.focus: str | None = None

    def register(self, identifier: str, proposed: dict, observations: list[dict]):
        self.records[identifier] = {
            "id": identifier,
            "status": "proposed",
            "phenomenon": proposed["concept"],
            "phenomenon_ids": proposed.get("phenomenon_ids", []),
            "mechanism": proposed["mechanism"],
            "rival": proposed["rival"],
            "falsifier": proposed["falsifier"],
            "measurement": proposed["measurement"],
            "scientific_contract": {
                key: proposed.get(key, "") for key in (
                    "target_reference", "reference_identification",
                    "distortion_quantity", "distortion_estimate",
                    "prediction_if_mechanism", "prediction_if_rival",
                    "affected_population", "mechanism_entry_point",
                    "reference_status", "next_information_needed",
                    "discriminating_experiment")},
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
            # Scientific evidence stage is independent of experiment execution.
            # Qwen can observe behavior without inventing a recommender Bias.
            "evidence_status": "behavioral_pattern",
            "bias_relevance": {
                "reference_kind": "none",
                "target_reference": proposed.get("target_reference", ""),
                "reference_identification": proposed.get("reference_identification", ""),
                "distortion_quantity": proposed.get("distortion_quantity", ""),
                "proposed_distortion": proposed.get("distortion_estimate", ""),
                "mechanism_entry_point": "",
                "corrective_intervention": "",
                "reference_measured": False,
            },
        }

    @staticmethod
    def check_prediction(choice: dict) -> None:
        """A single measurable contrast must discriminate the two hypotheses."""
        directions = {"increase", "decrease", "no_change"}
        first = choice["mechanism_direction"]
        rival = choice["rival_direction"]
        if first not in directions or rival not in directions or first == rival:
            raise ValueError("H1 and H2 predict the same measured direction; "
                             "choose a different subgroup, outcome, or contrast")
        if not all(str(choice[k]).strip() for k in (
            "contrast_group", "contrast_outcome", "contrast_source",
            "control_value", "treatment_value")):
            raise ValueError("Comparison needs actual group, outcome, source, and arms")
        if choice["control_value"] == choice["treatment_value"]:
            raise ValueError("Control and treatment labels must differ")

    def choose(self, identifier: str, rationale: str, next_query: str = ""):
        if identifier not in self.records:
            raise ValueError("Select an existing research opportunity")
        if self.records[identifier]["status"] in ("abandoned", "rejected_development", "design_blocked"):
            raise ValueError("This hypothesis has been abandoned after counterevidence; "
                             "select another mechanism or form a new concept")
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
        item["phenomenon"] = concept["concept"]
        item["mechanism"] = concept["mechanism"]
        item["rival"] = concept["rival"]
        item["falsifier"] = concept["falsifier"]
        item["measurement"] = concept["measurement"]
        if concept.get("query"):
            item["next_observation"] = concept["query"]
        for field in item["scientific_contract"]:
            if field in concept:
                item["scientific_contract"][field] = concept[field]
        if status in ("abandoned", "rejected_development", "design_blocked"):
            if self.focus == identifier:
                self.focus = None

    def record_research_decision(self, identifier: str, concept: dict, entry: dict):
        """Keep one evidence-linked next choice after execution and Reflection."""
        item = self.records[identifier]
        execution_failure = entry.get("tool_error")
        revision_failure = entry.get("reflection_veto")
        gaps = item.get("scientific_evidence_gaps", [])
        critique = concept.get("critique") or {}
        reflection = entry.get("reflection") or {}
        if concept["status"] in ("rejected_development", "design_blocked"):
            mode = "change_mechanism"
            reason = (execution_failure or revision_failure or
                      "The current mechanism was rejected or its design is blocked")
        elif execution_failure:
            mode = "repair_execution"
            reason = execution_failure
        elif revision_failure:
            mode = ("repair_execution" if "did not execute" in revision_failure
                    else "rethink_mechanism")
            reason = revision_failure
        elif reflection.get("decision") == "revise" and "revised_concept" in entry:
            mode = "test_revised_mechanism"
            reason = reflection["reason"]
        elif item.get("bias_relevance", {}).get("reference_spec", {}).get(
                "kind") == "controlled_model":
            # An owned-model causal test has a different next action from
            # diagnosing a production-platform reference. The latter remains
            # unidentified, but must not trap the scoped mechanism in SQL.
            trials = [t for t in item.get("controlled_reference_tests", [])
                      if t.get("tested_concept_version") == concept["version"]]
            h1 = any(t.get("prediction_audit", {}).get("supports") == "H1"
                     and t.get("prediction_audit", {}).get(
                         "distinguishes_predictions") for t in trials)
            if not trials:
                mode, reason = ("test_controlled_mechanism",
                                "Use the executable recommender to test this version")
            elif not h1:
                mode, reason = ("falsify_or_revise_mechanism",
                                "Controlled evidence did not distinguish H1 from the rival")
            elif concept.get("literature") is None:
                mode, reason = ("check_literature",
                                "H1 is supported inside the owned model; assess prior work")
            elif critique.get("decision") == "ready" and critique.get(
                    "observable_valid") and critique.get("rival_discriminated"):
                mode, reason = ("freeze_scoped_mechanism",
                                "Freeze for sealed confirmation; do not claim a platform Bias")
            else:
                mode, reason = ("reassess_controlled_mechanism",
                                "Resolve the critic's remaining scoped mechanism test")
        elif "numeric_reference_deviation_unmeasured" in gaps:
            mode = "identify_reference"
            reason = "No measured deviation from a defensible recommender target"
        elif "no_executed_inferential_rival_countertest" in gaps or (
                "competing_mechanism_not_discriminated" in gaps):
            mode = "disambiguate"
            reason = "H1/H2 has not been distinguished by numerical counterevidence"
        elif "recommendation_consequence_not_identified" in gaps:
            mode = "measure_consequence"
            reason = "Need recommender opportunity or utility consequence"
        else:
            mode = "investigate"
            reason = critique.get("reason", "Seek new evidence")
        item["next_research_decision"] = {
            "mode": mode,
            "reason": str(reason)[:300],
            "next_sql": (reflection.get("next_sql") if mode == "test_revised_mechanism"
                         else item.get("next_observation", ""))[:380],
            "reference_status": item.get("scientific_contract", {}).get(
                "reference_status", "unidentified"),
            "h1_h2_discriminated": bool(critique.get("rival_discriminated", False)
                                       and "competing_mechanism_not_discriminated" not in gaps),
        }

    def record_selection_design(self, identifier: str, choice: dict,
                                support: dict, capabilities: dict | None = None):
        from rsi_researcher.research.discovery.references import (
            ReferenceSpec, EstimandSpec)
        item = self.records[identifier]
        # A model-written reference is a proposal, not empirical confirmation.
        item["bias_relevance"] = {
            "reference_kind": choice.get("reference_kind", "none"),
            "target_reference": choice.get("target_reference", ""),
            "reference_identification": choice.get("reference_identification", ""),
            "proposed_distortion": choice.get("proposed_distortion", ""),
            "mechanism_entry_point": choice.get("mechanism_entry_point", ""),
            "corrective_intervention": choice.get("corrective_intervention", ""),
            "reference_measured": False,
        }
        # Reference assumptions originate with the Agent's scientific
        # hypothesis; opportunity selection must not silently erase them.
        reference_choice = {
            **choice,
            "reference_identification": (
                choice.get("reference_identification") or
                item["scientific_contract"].get("reference_identification", "")),
        }
        item["bias_relevance"]["reference_spec"] = ReferenceSpec.from_choice(
            reference_choice, capabilities or {}).as_record()
        item["evidence_status"] = "behavioral_pattern"
        item["selection_contract"] = {
            "estimand_spec": EstimandSpec.from_choice(choice).as_record(),
            "source_table": choice["contrast_source"],
            "group_expression": choice["contrast_group"],
            "outcome_expression": choice["contrast_outcome"],
            "population_filter": choice["contrast_filter"],
            "control_value": choice["control_value"],
            "treatment_value": choice["treatment_value"],
            "analysis": choice["analysis"],
            "prediction_if_mechanism": choice["mechanism_direction"],
            "prediction_if_rival": choice["rival_direction"],
            "development_support": support,
        }

    def record_reference_measurement(self, identifier: str, stats: dict, *, scope: str,
                                     executed_specification: dict | None = None,
                                     target_identity: dict | None = None):
        """Record an executed reference contrast without claiming a normative target.

        A paired-user mean difference need not equal a pooled-share difference.
        Both values are retained; scientific identification remains separate.
        """
        if identifier not in self.records:
            raise ValueError("Unknown research opportunity")
        required = ("estimate", "ci95_bootstrap", "mean_control", "mean_treatment")
        if not all(k in stats for k in required):
            raise ValueError("Reference measurement needs executed Workbench statistics")
        ci = stats["ci95_bootstrap"]
        values = [stats["estimate"], stats["mean_control"], stats["mean_treatment"], *ci]
        if len(ci) != 2 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Invalid measured reference gap or uncertainty")
        bias = self.records[identifier]["bias_relevance"]
        bias["numeric_discrepancy"] = {
            "metric": "user_paired_group_exposure_share_difference",
            "observed": stats["mean_treatment"],
            "reference": stats["mean_control"],
            "estimate": stats["estimate"],
            "ci95": ci,
            "measured": True,
            "reference_scope": scope,
            "executed_specification": executed_specification,
            "agent_selected_target_identity": target_identity,
            "normative_target_validated": False,
        }
        bias["reference_measured"] = True
        if "reference_spec" in bias:
            bias["reference_spec"]["status"] = "measured_unidentified"
        self.records[identifier]["evidence_status"] = "reference_diagnostic_only"

    def record_controlled_reference(self, identifier: str, evidence: dict,
                                    prediction_audit: dict, *, concept_version: int = 1,
                                    observational_anchor: dict | None = None):
        """Promote executed evidence only in its actual, constructed-model scope."""
        if identifier not in self.records:
            raise ValueError("Unknown research opportunity")
        item = self.records[identifier]
        bias = item["bias_relevance"]
        if bias.get("reference_spec", {}).get("kind") != "controlled_model":
            raise ValueError("Intervention was not selected against a controlled-model reference")
        stats = evidence["result"]
        design = evidence["identification"]
        estimand = evidence["estimand"]
        specification = evidence["specification"]
        treatment = estimand["treatment"]
        if (estimand["statistical_unit"] != "uid"
                or estimand["phase"] != "development"
                or estimand["quantity"] != "mean_per_user_treatment_minus_control"
                or estimand["outcome"] != specification["outcome"]
                or stats["outcome"] != specification["outcome"]
                or treatment != {
                    "parameter": specification["parameter"],
                    "control": specification["control_value"],
                    "intervention": specification["treatment_value"],
                }):
            raise ValueError("Executed controlled estimand differs from the registered design")
        ci = stats["ci95_bootstrap"]
        values = [stats["estimate"], stats["mean_control"],
                  stats["mean_treatment"], *ci]
        if (len(ci) != 2 or not all(math.isfinite(float(v)) for v in values)
                or not math.isclose(stats["mean_treatment"] - stats["mean_control"],
                                    stats["estimate"], abs_tol=1e-9)
                or stats["n_paired_units"] < 12
                or stats["negative_control_passed"] is False
                or (stats["negative_control_tested"] and
                    stats["negative_control_passed"] is not True)
                or not design["within_model_paired_intervention"]
                or not design["same_users"]
                or not design["same_candidates"]
                or not design["same_heldout_relevance_labels"]
                or not design["train_users_separate_from_dev"]
                or design["confirmation_users_consulted"]
                or design["platform_policy_identified"]):
            raise ValueError("Controlled reference lacks a valid independent paired comparison")
        bias["numeric_discrepancy"] = {
            "metric": stats["outcome"],
            "observed": stats["mean_control"],
            "reference": stats["mean_treatment"],
            "estimate": stats["estimate"],
            "ci95": ci,
            "measured": True,
            "reference_scope": evidence["evidence_scope"],
            "normative_target_validated": False,
            "controlled_model_effect_identified": True,
        }
        bias["reference_measured"] = True
        bias["reference_spec"]["status"] = "identified_model_effect_only"
        item["recommendation_consequence"] = {
            "identified": True,
            "metric": stats["outcome"],
            "estimate": stats["estimate"],
            "ci95": ci,
            "reference_scope": evidence["evidence_scope"],
            "platform_causal_effect": False,
        }
        item.setdefault("controlled_reference_tests", []).append({
            "tested_concept_version": concept_version,
            "estimand": estimand,
            "specification": specification,
            "statistics": stats,
            "identification": design,
            "prediction_audit": prediction_audit,
            "observational_anchor": observational_anchor or {},
        })
        item["evidence_status"] = "controlled_model_effect_identified"

    def invalidate_revised_reference(self, identifier: str):
        """Old model effects stay recorded, but cannot validate a changed estimand."""
        item = self.records[identifier]
        bias = item["bias_relevance"]
        if not item.get("controlled_reference_tests"):
            return
        bias.pop("numeric_discrepancy", None)
        bias["reference_measured"] = False
        if "reference_spec" in bias:
            bias["reference_spec"]["status"] = "revised_requires_retest"
        item["recommendation_consequence"] = {"identified": False}
        item["evidence_status"] = "revised_hypothesis_needs_retest"

    def record_reference_diagnostic(self, identifier: str, result: dict,
                                    executed_audit: dict | None = None):
        item = self.records[identifier]
        item["bias_relevance"]["randomized_reference_diagnostic"] = result
        if executed_audit is not None:
            item["bias_relevance"]["randomized_reference_audit"] = executed_audit
        # Programmatically audit the exact contrast. This is a difference
        # between observed exposure modes, NOT a justified optimal-policy target.
        shares = result["treatment_group_shares_standard_random"]
        gap = result["standard_minus_random_treatment_group_share"]
        ci = result["group_share_gap_ci95_user_bootstrap"]
        if (len(shares) != 2 or len(ci) != 2
                or not all(math.isfinite(float(v)) for v in [*shares, gap, *ci])
                or not math.isclose(shares[0] - shares[1], gap, abs_tol=1e-9)):
            raise ValueError("Random-reference gap and uncertainty are not numerically consistent")
        item["bias_relevance"]["numeric_discrepancy"] = {
            "metric": "treatment_item_group_exposure_share",
            "observed": shares[0], "reference": shares[1],
            "estimate": gap, "ci95": ci, "measured": True,
            "reference_scope": "random-insertion exposure mode only",
            "normative_target_validated": False}
        item["bias_relevance"]["randomized_response_observed"] = True
        item["bias_relevance"]["reference_measured"] = True
        if "reference_spec" in item["bias_relevance"]:
            item["bias_relevance"]["reference_spec"]["status"] = (
                "measured_restricted_reference_only"
                if executed_audit and executed_audit.get("matches_selected_observable")
                else "measured_unidentified")
        item["evidence_status"] = "reference_diagnostic_only"

    def snapshot(self) -> dict:
        return {"focus": self.focus, "opportunities": list(self.records.values())}

    def for_agent(self) -> dict:
        """Compact enough for Qwen without hiding the competing mechanisms."""
        return {"focus": self.focus, "opportunities": [
            {k: v for k, v in item.items() if k not in
             ("observed_evidence", "tests")} |
            {"observed_evidence": item["observed_evidence"][-2:],
             "controlled_reference_tests": item.get("controlled_reference_tests", [])[-1:],
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
