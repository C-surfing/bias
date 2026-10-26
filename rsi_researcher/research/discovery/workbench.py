"""Generic research experiment workbench, independent of named Bias families.

The researcher supplies variables and counterfactual contrasts; the tool owns
data access, statistical units, uncertainty and numeric auditing.
"""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from statistics import mean
from typing import Any

from rsi_researcher.research.discovery.open_research import FORBIDDEN

TEXT = {"type": "string"}
EXPERIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": TEXT,
        "source_table": {"type": "string", "enum": ["events", "features"]},
        "source_query": TEXT,
        "population_filter": TEXT,
        "unit_expression": TEXT,
        "group_expression": TEXT,
        "control_value": TEXT,
        "treatment_value": TEXT,
        "outcome_expression": TEXT,
        "strata_expression": TEXT,
        "analysis": {"type": "string", "enum": ["paired_user", "cluster_weighted"]},
        "placebo_source_query": TEXT,
        "placebo_group_expression": TEXT,
        "prediction": TEXT,
        "falsifier": TEXT,
        "meaningful_margin": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "question", "source_table", "source_query", "population_filter",
        "unit_expression", "group_expression", "control_value", "treatment_value",
        "outcome_expression", "strata_expression", "analysis",
        "placebo_source_query", "placebo_group_expression", "prediction",
        "falsifier", "meaningful_margin"],
    "additionalProperties": False,
}


def _expr(expr: str, *, can_be_empty: bool = False) -> str:
    exp = expr.strip()
    if not exp and can_be_empty:
        return ""
    if not exp or len(exp) > 2000 or ";" in exp or FORBIDDEN.search(exp):
        raise ValueError("Invalid research SQL expression")
    return exp


def _source(sql: str, table: str) -> str:
    if not sql.strip():
        return table
    statement = sql.strip().rstrip(";")
    if (not re.match(r"^(SELECT|WITH)\b", statement, re.I)
        or ";" in statement or len(statement) > 4000 or FORBIDDEN.search(statement)):
        raise ValueError("source_query must be a single SELECT/WITH")
    return "(" + statement + ")"


def _summary(rows: list[dict], spec: dict, *, seed: int) -> dict:
    # Result rows are per-(user, group, stratum) sufficient statistics.
    control, treatment = spec["control_value"], spec["treatment_value"]
    if control == treatment:
        raise ValueError("Control and treatment labels must differ")
    user_data: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0.]))
    by_stratum: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0.]))
    totals = {control: [0, 0.], treatment: [0, 0.]}
    for item in rows:
        group = item["grp"]
        if group not in totals:
            continue
        count, value = int(item["n"]), float(item["sy"])
        if count <= 0 or not math.isfinite(value):
            continue
        unit = str(item["unit"])
        user_data[unit][group][0] += count
        user_data[unit][group][1] += value
        by_stratum[str(item["stratum"])][group][0] += count
        by_stratum[str(item["stratum"])][group][1] += value
        totals[group][0] += count
        totals[group][1] += value
    if any(totals[g][0] < 5 for g in totals):
        raise ValueError("Both comparison groups need at least five observed outcomes")
    paired = {
        unit: values for unit, values in user_data.items()
        if all(values[group][0] > 0 for group in (control, treatment))
    }
    if len(paired) < 5:
        raise ValueError("Too few independent users with both comparison groups")
    def contrast(values: dict) -> float:
        return (values[treatment][1] / values[treatment][0]
                - values[control][1] / values[control][0])
    analysis = spec["analysis"]
    if analysis == "paired_user":
        observed = mean(contrast(value) for value in paired.values())
    else:
        observed = contrast(totals)
    rng = random.Random(seed)
    users = list(user_data)
    if analysis == "paired_user":
        per_user = [contrast(v) for v in paired.values()]
        estimates = [mean(rng.choices(per_user, k=len(per_user)))
                     for _ in range(200)]
    else:
        estimates = []
        for _ in range(200):
            sampled = rng.choices(users, k=len(users))
            sums = {control: [0, 0.], treatment: [0, 0.]}
            for uid in sampled:
                for grp in (control, treatment):
                    n, sy = user_data[uid][grp]
                    sums[grp][0] += n
                    sums[grp][1] += sy
            if all(sums[group][0] for group in sums):
                estimates.append(contrast(sums))
    if len(estimates) < 100:
        raise ValueError("Insufficient valid user-cluster bootstrap replicates")
    estimates.sort()
    lower, upper = estimates[int(.025 * len(estimates))], estimates[int(.975 * len(estimates)) - 1]
    strata = {}
    for name, values in by_stratum.items():
        if all(values[group][0] >= 5 for group in (control, treatment)):
            strata[name] = {
                "effect": contrast(values),
                "n_control": values[control][0],
                "n_treatment": values[treatment][0]}
    return {
        "estimate": observed, "ci95_bootstrap": [lower, upper],
        "analysis_unit": "uid_or_agent_defined_unit",
        "analysis": analysis,
        "n_independent_units": len(user_data),
        "n_paired_units": len(paired),
        "n_control": totals[control][0], "n_treatment": totals[treatment][0],
        "mean_control": totals[control][1] / totals[control][0],
        "mean_treatment": totals[treatment][1] / totals[treatment][0],
        "strata": strata,
        "bootstrap_replicates": len(estimates),
        "ci_inside_declared_margin": (lower > -spec["meaningful_margin"]
                                      and upper < spec["meaningful_margin"]
                                      if spec["meaningful_margin"] > 0 else None),
    }


class ExperimentalWorkbench:
    """Execute a complete Experiment Specification on the *development* handle."""

    def __init__(self, development_tools):
        self.tools = development_tools

    def _measure(self, spec: dict, *, placebo: bool = False) -> dict:
        source_query = (spec["placebo_source_query"] if placebo and
                        spec["placebo_source_query"].strip() else spec["source_query"])
        if source_query and re.search(r"\bGROUP\s+BY\b|\bAVG\s*\(|\bCOUNT\s*\(", source_query, re.I):
            raise ValueError(
                "source_query must preserve per-event rows and independent unit IDs; "
                "do NOT pre-aggregate. Use source_query='' and outcome_expression='click' "
                "(or another raw outcome). Workbench performs the user-level aggregation.")
        source = _source(source_query, spec["source_table"])
        group = (spec["placebo_group_expression"] if placebo else spec["group_expression"])
        if placebo and not group.strip():
            raise ValueError("Placebo must have a different group expression")
        outcome, unit = _expr(spec["outcome_expression"]), _expr(spec["unit_expression"])
        condition = spec["population_filter"].strip()
        # A frequent LLM formatting mistake is writing the WHERE keyword in a
        # field whose semantic content is only the boolean population condition.
        condition = re.sub(r"^WHERE\s+", "", condition, flags=re.I)
        where = _expr(condition) if condition else "TRUE"
        group = _expr(group)
        stratum = _expr(spec["strata_expression"]) if spec["strata_expression"].strip() else "'all'"
        # Only raw development tables or a model-generated SELECT on that database.
        query = (
            "WITH base AS (SELECT CAST((" + unit + ") AS VARCHAR) unit,"
            " CAST((" + group + ") AS VARCHAR) grp,"
            " CAST((" + outcome + ") AS DOUBLE) y,"
            " CAST((" + stratum + ") AS VARCHAR) stratum"
            " FROM " + source + " AS observations WHERE (" + where + ")),"
            " per_user AS (SELECT unit,grp,stratum,COUNT(*) n,SUM(y) sy"
            " FROM base WHERE y IS NOT NULL AND isfinite(y)"
            " AND grp IN ('" + _safe_literal(spec["control_value"]) + "','"
            + _safe_literal(spec["treatment_value"]) + "')"
            " GROUP BY 1,2,3)"
            " SELECT * FROM per_user ORDER BY unit,grp,stratum")
        # Our existing SQL accessor caps rows at 30; use its read-only connection for
        # per-unit sufficient stats (not raw trajectories), then log the full contract.
        from rsi_researcher.research.discovery.open_research import FORBIDDEN as _forbidden
        # query is assembled from the validated research plan; outer database has
        # enable_external_access=false and is opened read_only.
        if _forbidden.search(query):
            raise ValueError("Unsupported SQL operation")
        result = self.tools.connection.execute(query)
        header = [d[0] for d in result.description]
        records = [dict(zip(header, row)) for row in result.fetchall()]
        if len(records) > 60000:
            raise ValueError("Experiment grouping cardinality too high")
        return {"sql": query, "sufficient_statistics": records}

    def run(self, spec: dict) -> dict:
        from jsonschema import Draft202012Validator
        Draft202012Validator(EXPERIMENT_SCHEMA).validate(spec)
        # Models sometimes provide a SQL-quoted literal instead of its value;
        # stripping only *matching outer quotes* does not change the group.
        spec = dict(spec)
        for key in ("control_value", "treatment_value"):
            label = spec[key].strip()
            if len(label) >= 2 and label.startswith("'") and label.endswith("'"):
                label = label[1:-1]
            spec[key] = label
        source = self._measure(spec)
        effect = _summary(source["sufficient_statistics"], spec, seed=20261024)
        placebo = None
        if spec["placebo_group_expression"].strip():
            if (spec["placebo_group_expression"].strip() == spec["group_expression"].strip()
                and spec["placebo_source_query"].strip() == spec["source_query"].strip()):
                raise ValueError(
                    "A negative control cannot reuse the exact same treatment and source. "
                    "Construct a different group or derived source, or leave placebo fields empty.")
            placebo_stats = self._measure(spec, placebo=True)
            placebo = _summary(placebo_stats["sufficient_statistics"], spec, seed=20261025)
            placebo = {
                "result": placebo,
                "difference_from_placebo": effect["estimate"] - placebo["estimate"],
                "executed_sql": placebo_stats["sql"]}
        # Return compact evidence to the LLM, keep full reproducibility in journal.
        return {
            "specification": spec,
            "result": effect, "negative_control": placebo,
            "executed_sql": source["sql"],
            "scientific_status": "development_only_not_independent_confirmation",
            "identifies_causal_bias": False,
            "columns": ["estimate", "ci_low", "ci_high", "n_users"],
            "rows": [{"estimate": effect["estimate"],
                      "ci_low": effect["ci95_bootstrap"][0],
                      "ci_high": effect["ci95_bootstrap"][1],
                      "n_users": effect["n_independent_units"]}],
            "sql": source["sql"],
        }


def _safe_literal(value: str) -> str:
    if len(value) > 100 or any(x in value for x in ("'", ";", "\\", "\n")):
        raise ValueError("Group label contains unsafe characters")
    return value
