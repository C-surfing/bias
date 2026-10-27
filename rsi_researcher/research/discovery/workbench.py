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
        support = ", ".join(f"{key}: {values[0]} events" for key, values in totals.items())
        raise ValueError("Insufficient observed support for comparison: " + support +
                         ". Inspect the actual group distribution and use supported values.")
    paired = {
        unit: values for unit, values in user_data.items()
        if all(values[group][0] > 0 for group in (control, treatment))
    }
    if spec["analysis"] == "paired_user" and len(paired) < 5:
        raise ValueError("Too few independent users with both comparison groups")
    if spec["analysis"] == "cluster_weighted" and any(
        sum(values[arm][0] > 0 for values in user_data.values()) < 5
        for arm in (control, treatment)
    ):
        raise ValueError("Cluster-weighted inference needs at least five independent users per group")
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

    def supported_population_filters(self, spec: dict) -> list[dict]:
        """Observe support, not outcomes, for alternative experiment cohorts."""
        if spec["source_query"].strip():
            return []
        names = [item["column"] for item in self.tools.schema()[spec["source_table"]]]
        excluded = {"uid", "vid", "click", "hate", "long_view", "next_click",
                    "moment_ms", "timestamp", "previous_moment"}
        source = self.tools.active_table(spec["source_table"])
        condition = re.sub(r"^WHERE\s+", "", spec["population_filter"].strip(), flags=re.I)
        where = _expr(condition) if condition else "TRUE"
        group = _expr(spec["group_expression"])
        result = []
        for col in names:
            if col in excluded or not re.fullmatch(r"[a-z][a-z0-9_]*", col):
                continue
            try:
                rows = self.tools.connection.execute(
                    "SELECT " + col + ", CAST((" + group + ") AS VARCHAR),"
                    " COUNT(DISTINCT uid) FROM " + source + " AS observations"
                    " WHERE (" + where + ") GROUP BY 1,2").fetchall()
            except Exception:
                continue
            levels = sorted({row[0] for row in rows if row[0] is not None}, key=str)
            if not (2 <= len(levels) <= 8):
                continue
            for values in (levels[:len(levels)//2], levels[len(levels)//2:]):
                by_arm = {str(spec["control_value"]): 0, str(spec["treatment_value"]): 0}
                for value, arm, count in rows:
                    if value in values and arm in by_arm:
                        by_arm[arm] += int(count)
                if min(by_arm.values()) < 5:
                    continue
                literals = [str(v) if isinstance(v, (int, float)) else
                            "'" + str(v).replace("'", "''") + "'" for v in values]
                clause = col + " IN (" + ",".join(literals) + ")"
                proposal = ("(" + where + ") AND (" + clause + ")"
                            if where.upper() != "TRUE" else clause)
                if proposal != spec["population_filter"]:
                    result.append({"population_filter": proposal,
                                   "support_users_by_arm": by_arm})
            if len(result) >= 6:
                break
        return result[:6]

    def observed_strata_columns(self, table: str) -> list[str]:
        """Discover cheap, low-cardinality pre-outcome strata from actual rows.

        No bias categories or experimental questions are hard-coded here.
        """
        columns = [
            c["column"] for c in self.tools.schema()[table]
            if re.fullmatch(r"[a-z][a-z0-9_]*", c["column"])
            and c["column"] not in (
                "uid", "vid", "click", "hate", "long_view", "next_click")]
        columns.extend(self.tools.features.get(table, {}))
        if not columns:
            return [""]
        source = self.tools.active_table(table)
        query = ("SELECT " + ", ".join(
            "approx_count_distinct(" + field + ")" for field in columns) +
            " FROM (SELECT * FROM " + source + " LIMIT 20000)")
        counts = self.tools.connection.execute(query).fetchone()
        return [""] + [
            field for field, count in zip(columns, counts)
            if 2 <= count <= 32]

    def inspect_plan(self, spec: dict) -> dict:
        """Check the *proposed* estimand on development rows before inference.

        All counts and types come from DuckDB, not from a model prediction.
        Group sample support refers to distinct users (not repeated events).
        """
        from jsonschema import Draft202012Validator
        Draft202012Validator(EXPERIMENT_SCHEMA).validate(spec)
        if spec["control_value"] == spec["treatment_value"]:
            raise ValueError("Control and treatment values must differ")
        if spec["source_query"].strip():
            if re.search(r"\bGROUP\s+BY\b|\bAVG\s*\(|\bCOUNT\s*\(", spec["source_query"], re.I):
                raise ValueError("source_query must preserve per-event rows and independent unit IDs")
            source = self.tools.rewrite_sources(_source(spec["source_query"], spec["source_table"]))
        else:
            source = self.tools.active_table(spec["source_table"])
        outcome = _expr(spec["outcome_expression"])
        if re.search(r"\b(AVG|COUNT|SUM|MIN|MAX)\s*\(|\bAS\b|,", outcome, re.I):
            raise ValueError("Use one numeric per-event outcome; Workbench handles aggregation")
        where = re.sub(r"^WHERE\s+", "", spec["population_filter"].strip(), flags=re.I)
        where = _expr(where) if where else "TRUE"
        group = _expr(spec["group_expression"])
        uid = _expr(spec["unit_expression"])
        dtype = self.tools.connection.execute(
            "SELECT typeof((" + outcome + ")) FROM " + source +
            " AS observations LIMIT 1").fetchone()
        dtype = dtype[0].upper() if dtype else "UNKNOWN"
        numeric = ("INTEGER", "INT", "BIGINT", "SMALLINT", "TINYINT",
                   "HUGEINT", "UBIGINT", "USMALLINT", "UTINYINT",
                   "DOUBLE", "FLOAT", "REAL", "DECIMAL", "BOOLEAN")
        if not dtype.startswith(numeric):
            available = []
            for field in self.tools.schema()[spec["source_table"]]:
                if field.get("type", "").upper().startswith(numeric):
                    available.append(field["column"])
            raise ValueError("Outcome '" + outcome + "' has type " + dtype +
                             ", not a numeric per-event outcome. Actual numeric columns: " +
                             ", ".join(available[:18]))
        labels = (spec["control_value"].strip("'"), spec["treatment_value"].strip("'"))
        if any(len(label) > 100 or "'" in label for label in labels):
            raise ValueError("Unsupported comparison labels")
        sql = (
            "WITH observed AS (SELECT CAST((" + uid + ") AS VARCHAR) u,"
            " CAST((" + group + ") AS VARCHAR) arm,"
            " CAST((" + outcome + ") AS DOUBLE) y"
            " FROM " + source + " AS observations WHERE (" + where + "))"
            " SELECT u,"
            " SUM(CASE WHEN arm='" + labels[0] + "' THEN 1 ELSE 0 END) n_control,"
            " SUM(CASE WHEN arm='" + labels[1] + "' THEN 1 ELSE 0 END) n_treatment"
            " FROM observed WHERE y IS NOT NULL AND isfinite(y)"
            " AND arm IN ('" + labels[0] + "','" + labels[1] + "')"
            " GROUP BY u"
        )
        groups = self.tools.connection.execute(sql).fetchall()
        control = sum(c > 0 for _, c, t in groups)
        treatment = sum(t > 0 for _, c, t in groups)
        paired = sum(c > 0 and t > 0 for _, c, t in groups)
        valid = paired >= 5 if spec["analysis"] == "paired_user" else min(control, treatment) >= 5
        return {
            "outcome_type": dtype, "outcome_numeric": True,
            "n_control_users": control, "n_treatment_users": treatment,
            "n_paired_users": paired,
            "n_control_events": sum(c for _, c, _ in groups),
            "n_treatment_events": sum(t for _, _, t in groups),
            "analysis": spec["analysis"],
            "valid_support": valid,
            "low_information_warning": (paired < 20 if spec["analysis"] == "paired_user"
                                        else min(control, treatment) < 20),
        }

    def _measure(self, spec: dict, *, placebo: bool = False) -> dict:
        source_query = (spec["placebo_source_query"] if placebo and
                        spec["placebo_source_query"].strip() else spec["source_query"])
        if source_query and re.search(r"\bGROUP\s+BY\b|\bAVG\s*\(|\bCOUNT\s*\(", source_query, re.I):
            raise ValueError(
                "source_query must preserve per-event rows and independent unit IDs; "
                "do NOT pre-aggregate. Use source_query='' and outcome_expression='click' "
                "(or another raw outcome). Workbench performs the user-level aggregation.")
        source = (self.tools.rewrite_sources(_source(source_query, spec["source_table"]))
                  if source_query.strip() else self.tools.active_table(spec["source_table"]))
        group = (spec["placebo_group_expression"] if placebo else spec["group_expression"])
        if placebo and not group.strip():
            raise ValueError("Placebo must have a different group expression")
        outcome, unit = _expr(spec["outcome_expression"]), _expr(spec["unit_expression"])
        if re.search(r"\b(AVG|COUNT|SUM|MIN|MAX)\s*\(|\bAS\b|,", outcome, re.I):
            raise ValueError("outcome_expression must be ONE per-event numeric scalar "
                             "(such as click or long_view), not AVG(...), aliases, "
                             "multiple selected expressions, or an aggregate; "
                             "Workbench calculates per-user averages itself")
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

    def run(self, spec: dict, *, required_feature: str | None = None) -> dict:
        from jsonschema import Draft202012Validator
        Draft202012Validator(EXPERIMENT_SCHEMA).validate(spec)
        if required_feature is not None:
            origin = self.tools.feature_records[required_feature]["source_table"]
            if spec["source_table"] != origin:
                raise ValueError("Registered feature source mismatch: " + required_feature +
                                 " originates in " + origin)
            fields = ("source_query", "population_filter", "group_expression",
                      "outcome_expression", "strata_expression")
            if not any(re.search(r"\b" + re.escape(required_feature) + r"\b", spec[f])
                       for f in fields):
                raise ValueError("Use registered feature name '" + required_feature +
                                 "' in the executable Experiment Specification; "
                                 "do not repeat its SQL definition")
        # Models sometimes provide a SQL-quoted literal instead of its value;
        # stripping only *matching outer quotes* does not change the group.
        spec = dict(spec)
        for key in ("control_value", "treatment_value"):
            label = spec[key].strip()
            if len(label) >= 2 and label.startswith("'") and label.endswith("'"):
                label = label[1:-1]
            spec[key] = label
        feasibility = self.inspect_plan(spec)
        if not feasibility["valid_support"]:
            raise ValueError("Insufficient independent user support: " + str(feasibility))
        source = self._measure(spec)
        effect = _summary(source["sufficient_statistics"], spec, seed=20261024)
        placebo = None
        omitted_placebo = None
        if (spec["placebo_group_expression"].strip() == spec["group_expression"].strip()
                and spec["placebo_source_query"].strip() == spec["source_query"].strip()):
            # A copied treatment is not an independent negative control.
            # Ignore the invalid optional placebo explicitly; never credit it.
            omitted_placebo = "Identical treatment and placebo were not tested as a negative control"
            spec["placebo_group_expression"] = ""
            spec["placebo_source_query"] = ""
        if spec["placebo_group_expression"].strip():
            placebo_stats = self._measure(spec, placebo=True)
            placebo = _summary(placebo_stats["sufficient_statistics"], spec, seed=20261025)
            placebo = {
                "result": placebo,
                "difference_from_placebo": effect["estimate"] - placebo["estimate"],
                "executed_sql": placebo_stats["sql"]}
        # Return compact evidence to the LLM, keep full reproducibility in journal.
        return {
            "specification": spec,
            "feasibility": feasibility,
            "result": effect, "negative_control": placebo,
            "omitted_invalid_placebo": omitted_placebo,
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



def randomized_reference_probe(tools, choice: dict) -> dict:
    """Development-only diagnostic, NOT identification of platform bias.

    Separately estimate the response contrast on randomly inserted exposures
    and the difference in observed group composition between standard and random logs.
    Restrict both policies to their observed overlapping calendar days.
    Unit bootstrap is by user, not by events.  A randomized insertion is
    not a randomized recommender policy or a normative optimal allocation.
    """
    import random
    from collections import defaultdict

    if choice["contrast_source"] != "events" or choice.get("reference_kind") != "randomized_exposure":
        raise ValueError("Randomized reference needs events and an explicitly selected randomized-exposure target")
    group = _expr(choice["contrast_group"])
    outcome = _expr(choice["contrast_outcome"])
    where = _expr(choice["contrast_filter"]) if choice["contrast_filter"].strip() else "TRUE"
    arms = (choice["control_value"], choice["treatment_value"])
    if arms[0] == arms[1]:
        raise ValueError("Identical comparison arms")
    for arm in arms:
        _safe_literal(arm)
    days = [row[0] for row in tools.connection.execute(
        "SELECT event_day FROM events GROUP BY event_day "
        "HAVING COUNT(*) FILTER (WHERE is_random=1)>0 "
        "AND COUNT(*) FILTER (WHERE is_random=0)>0 ORDER BY event_day").fetchall()]
    if not days:
        raise ValueError("No calendar days with both exposure policies")
    source = tools.active_table("events")
    days_sql = ",".join(str(int(d)) for d in days)
    query = (
        "SELECT CAST(uid AS VARCHAR) u, CAST(is_random AS INTEGER) exposure_mode,"
        "CAST((" + group + ") AS VARCHAR) arm,"
        "COUNT(*) n, SUM(CAST((" + outcome + ") AS DOUBLE)) sy "
        "FROM " + source + " WHERE event_day IN (" + days_sql + ") "
        "AND (" + where + ") AND (" + outcome + ") IS NOT NULL "
        "AND CAST((" + group + ") AS VARCHAR) IN ('" + _safe_literal(arms[0]) + "','" + _safe_literal(arms[1]) + "') "
        "GROUP BY 1,2,3")
    rows = tools.connection.execute(query).fetchall()
    per_user = defaultdict(lambda: {(0, arms[0]): (0, 0.0),
                                    (0, arms[1]): (0, 0.0),
                                    (1, arms[0]): (0, 0.0),
                                    (1, arms[1]): (0, 0.0)})
    for uid, policy, arm, n, sy in rows:
        if policy not in (0, 1) or arm not in arms or sy is None:
            continue
        per_user[uid][(policy, arm)] = (int(n), float(sy))
    users = list(per_user)
    if not users:
        raise ValueError("No events for the proposed random reference")
    def calc(cohort):
        aggregate = {(p,a): [0,0.] for p in (0,1) for a in arms}
        for uid in cohort:
            for key, (n,sy) in per_user[uid].items():
                aggregate[key][0] += n
                aggregate[key][1] += sy
        if min(aggregate[(p,a)][0] for p in (0,1) for a in arms) < 20:
            return None
        rnd_rates = [aggregate[(1,a)][1] / aggregate[(1,a)][0] for a in arms]
        std_rates = [aggregate[(0,a)][1] / aggregate[(0,a)][0] for a in arms]
        rnd_gap = rnd_rates[1] - rnd_rates[0]
        std_gap = std_rates[1] - std_rates[0]
        shares = []
        for policy in (0,1):
            denom = sum(aggregate[(policy,a)][0] for a in arms)
            if denom == 0:
                return None
            shares.append(aggregate[(policy,arms[1])][0] / denom)
        return (rnd_gap, shares[0]-shares[1], std_gap, std_gap-rnd_gap,
                rnd_rates, std_rates, shares, aggregate)
    point = calc(users)
    if point is None:
        raise ValueError("Insufficient randomized observations of both arms")
    rnd_users_by_arm = {
        arm: sum(per_user[u][(1,arm)][0]>0 for u in users) for arm in arms}
    if min(rnd_users_by_arm.values()) < 10:
        raise ValueError("Too few independent random-reference users per arm: "
                         + str(rnd_users_by_arm))
    rng = random.Random(20261026)
    boot = []
    for _ in range(240):
        sample = rng.choices(users, k=len(users))
        estimate = calc(sample)
        if estimate is not None:
            boot.append(estimate[:4])
    if len(boot) < 190:
        raise ValueError("Insufficient user-bootstrap reference support")
    def ci(i):
        vals = sorted(v[i] for v in boot)
        return [vals[int(.025*len(vals))], vals[int(.975*len(vals))-1]]
    response, allocation, standard_response, interaction, rates, standard_rates, shares, counts = point
    return {
        "status": "development_randomized_reference_diagnostic",
        "reference": "randomly inserted item response; not randomized full recommender",
        "same_calendar_days": days,
        "source_table": "events",
        "group_expression": group,
        "outcome_expression": outcome,
        "control_value": arms[0], "treatment_value": arms[1],
        "population_filter": where,
        "random_response_contrast_treatment_minus_control": response,
        "random_response_ci95_user_bootstrap": ci(0),
        "random_response_rates_control_treatment": rates,
        "standard_response_contrast_treatment_minus_control": standard_response,
        "standard_response_ci95_user_bootstrap": ci(2),
        "standard_response_rates_control_treatment": standard_rates,
        "standard_minus_random_response_contrast": interaction,
        "policy_response_interaction_ci95_user_bootstrap": ci(3),
        "standard_minus_random_treatment_group_share": allocation,
        "group_share_gap_ci95_user_bootstrap": ci(1),
        "treatment_group_shares_standard_random": shares,
        "n_independent_users": len(users),
        "n_random_users_by_arm": rnd_users_by_arm,
        "counts_by_policy_arm": {
            ("random" if p else "standard") + "_" + ("control" if a==arms[0] else "treatment"):
            counts[(p,a)][0] for p in (0,1) for a in arms},
        "response_vs_composition_sign_disagreement": bool(response*allocation < 0),
        "discrepancy_is_validated_bias": False,
        "limitations": (
            "Group shares can reflect user/history state rather than item allocations; ""even item group shares need an independently justified utility target; "
            "random insertion does not randomize the learned recommender, "
            "user state, item mix or policy eligibility. "
            "This contrast does not establish causal algorithmic harm."),
    }
