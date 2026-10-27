"""Model-directed, evidence-first research loop on physically isolated development data.

Exploration can define variables, inspect trajectories, form and revise concepts,
run discrimination tests, consult literature, and freeze an independently testable
proposal. Only the evaluator (outside this module) may touch holdout.
"""
from __future__ import annotations

import json
import itertools
import re
import urllib.parse
import urllib.request
from pathlib import Path

from rsi_researcher.research.discovery.opportunities import ResearchOpportunityPool
from rsi_researcher.research.scientific_agents import ResearchAgent
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.client import BudgetExceeded, ModelFailure
from rsi_researcher.runtime.models import create_model

BIAS_RESEARCH_CONTRACT = (
    "Shared recommender bias research contract. Study a specific RECOMMENDER "
    "decision, exposure allocation, data collection, preference learning, or "
    "feedback update relative to an identified and defensible target. "
    "A difference in click rates, user preferences, tag frequency or activity "
    "is a behavioral observation, NOT automatically algorithmic distortion. "
    "Name the affected population, distorted quantity, decision stage, "
    "reference and its identification assumptions. Randomly inserted KuaiRand "
    "items provide a limited exposure-mode response reference, NOT randomized "
    "user histories, known utility merit, or a valid policy counterfactual. "
    "Missing references are UNMEASURED or UNIDENTIFIED research gaps: "
    "investigate support rather than invent targets or abandon promising leads. "
    "Derive competing H1 and H2 predictions on the same population, estimand "
    "and conditioning variables; choose an informative contrast where they "
    "disagree, not a test maximizing significance. "
    "Distinguish observing a phenomenon, measuring reference deviation, "
    "discriminating a mechanism, and testing a corrective intervention. "
    "Use measured data and tool uncertainty only. Diagnose whether failure "
    "comes from invalid reference, unsupported mechanism, insufficient "
    "identification/support or an invalid experiment. Do not claim causal "
    "bias, harm, novelty or independent confirmation from exploratory logs. "
)

TEXT = {"type": "string"}
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [
            "sql", "inspect_trajectory", "define_feature", "form_concept",
            "select_opportunity", "test", "experiment", "revise", "literature",
            "freeze", "stop"]},
        "reason": TEXT, "query": TEXT, "feature_name": TEXT, "expression": TEXT,
        "source_table": {"type": "string", "enum": ["features", "events"]},
        "concept_id": TEXT, "concept": TEXT, "mechanism": TEXT, "rival": TEXT,
        "falsifier": TEXT, "measurement": TEXT,
    },
    "required": ["action", "reason", "query", "feature_name", "expression", "source_table",
                 "concept_id", "concept", "mechanism", "rival", "falsifier", "measurement"],
    "additionalProperties": False,
}
CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["test", "revise", "reject", "ready"]},
        "reason": TEXT, "next_sql": TEXT,
        "observable_valid": {"type": "boolean"},
        "rival_discriminated": {"type": "boolean"},
        "logging_artifact_excluded": {"type": "boolean"},
    },
    "required": ["decision", "reason", "next_sql", "observable_valid",
                 "rival_discriminated", "logging_artifact_excluded"],
    "additionalProperties": False,
}
CONCEPT_PAIR_SCHEMA = {
    "type": "object",
    "properties": {"concepts": {"type": "array", "minItems": 2, "maxItems": 3,
        "items": {"type": "object", "properties": {
            "concept": TEXT, "mechanism": TEXT, "rival": TEXT,
            "falsifier": TEXT, "measurement": TEXT, "suggested_sql": TEXT,
            "target_reference": TEXT, "reference_identification": TEXT,
            "distortion_quantity": TEXT, "distortion_estimate": TEXT,
            "prediction_if_mechanism": TEXT, "prediction_if_rival": TEXT,
            "affected_population": TEXT, "mechanism_entry_point": TEXT,
            "reference_status": {"type": "string", "enum": [
                "unmeasured", "unidentified", "identified"]},
            "next_information_needed": TEXT, "discriminating_experiment": TEXT},
            "required": ["concept", "mechanism", "rival", "falsifier",
                         "measurement", "suggested_sql"],
            "additionalProperties": False}}},
    "required": ["concepts"], "additionalProperties": False,
}
OPPORTUNITY_CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_concept_id": TEXT,
        "reason": TEXT,
        "discriminating_prediction": TEXT,
        "prediction_if_mechanism": TEXT,
        "prediction_if_rival": TEXT,
        "why_not_other": TEXT,
        "next_action": {"type": "string", "enum": ["observe", "experiment"]},
        "next_sql": TEXT,
        "experimental_cost": TEXT,
        "contrast_source": {"type": "string", "enum": ["events", "features"]},
        "contrast_group": TEXT,
        "contrast_outcome": TEXT,
        "contrast_filter": TEXT,
        "control_value": TEXT,
        "treatment_value": TEXT,
        "analysis": {"type": "string", "enum": ["paired_user", "cluster_weighted"]},
        "mechanism_direction": {"type": "string", "enum": [
            "increase", "decrease", "no_change"]},
        "rival_direction": {"type": "string", "enum": [
            "increase", "decrease", "no_change"]},
        "reference_kind": {"type": "string", "enum": [
            "none", "randomized_exposure", "dense_relevance", "controlled_model"]},
        "target_reference": TEXT,
        "proposed_distortion": TEXT,
        "mechanism_entry_point": TEXT,
        "corrective_intervention": TEXT,
    },
    "required": ["selected_concept_id", "reason", "discriminating_prediction",
                 "prediction_if_mechanism", "prediction_if_rival",
                 "why_not_other", "next_action", "next_sql", "experimental_cost",
                 "contrast_source", "contrast_group", "contrast_outcome",
                 "contrast_filter", "control_value", "treatment_value",
                 "analysis", "mechanism_direction", "rival_direction",
                 "reference_kind", "target_reference", "proposed_distortion",
                 "mechanism_entry_point", "corrective_intervention"],
    "additionalProperties": False,
}

REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["revise", "abandon", "literature"]},
        "reason": TEXT, "concept": TEXT, "mechanism": TEXT, "rival": TEXT,
        "falsifier": TEXT, "measurement": TEXT, "next_sql": TEXT,
    },
    "required": ["decision", "reason", "concept", "mechanism", "rival",
                 "falsifier", "measurement", "next_sql"],
    "additionalProperties": False,
}
COUNTER_DELTA_SCHEMA = {
    "type": "object",
    "properties": {
        "field": {"type": "string", "enum": [
            "population_filter", "group_expression",
            "outcome_expression"]},
        "replacement": {"type": "string", "minLength": 1},
        "reason": {"type": "string", "minLength": 8},
    },
    "required": ["field", "replacement", "reason"],
    "additionalProperties": False,
}
QUERY_SCHEMA = {"type": "object", "properties": {
    "search_queries": {"type": "array", "items": TEXT, "minItems": 2, "maxItems": 3}
}, "required": ["search_queries"], "additionalProperties": False}
NOVELTY_SCHEMA = {"type": "object", "properties": {
    "overlap": TEXT, "unresolved_difference": TEXT,
    "strongest_prior_work_ids": {"type": "array", "items": TEXT, "maxItems": 4},
    "verdict": {"type": "string", "enum": [
        "possibly_distinct", "likely_known", "insufficient_literature"]},
    "discriminating_next_test": TEXT,
}, "required": ["overlap", "unresolved_difference", "strongest_prior_work_ids",
               "verdict", "discriminating_next_test"], "additionalProperties": False}
FORBIDDEN = re.compile(
    r"\b(attach|detach|copy|create|drop|alter|insert|update|delete|replace|"
    r"pragma|call|load|install|export|import|read_csv|read_parquet|read_text|"
    r"sqlite_scan|postgres_scan|httpfs|glob|writefile|readfile|"
    r"duckdb_tables|duckdb_columns|information_schema|pg_catalog|"
    r"current_setting|set|execute|prepare|describe)\b", re.I)
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{2,40}$")


def development_db(source_db: str, target_db: str) -> dict:
    """Materialize development-fold rows physically, excluding confirmation users."""
    import duckdb
    dst = Path(target_db)
    if dst.exists():
        return {"path": str(dst), "reused": True}
    dst.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(dst))
    try:
        source = str(Path(source_db).resolve()).replace("'", "''")
        con.execute("ATTACH '" + source + "' AS source_db (READ_ONLY)")
        con.execute("""CREATE TABLE features AS SELECT
            uid,vid,event_day,moment_ms,source_file,activity,tags,
            click,long_view,next_click,after_long_gap,after_nonclick,
            after_negative_feedback,immediate_repeat,topic_changed,
            recent_response_drop,previous_interval_ms,recent_click,older_click
            FROM source_db.anomaly_features
            WHERE user_fold<>0 AND event_day<=20220430""")
        source_fields = {field[0] for field in con.execute(
            "DESCRIBE SELECT * FROM source_db.observed_sequences").fetchall()}
        optional = [field for field in (
            "author", "tab", "activity", "sequence_index",
            "previous_long_view", "recent_click", "older_click")
            if field in source_fields]
        extras = "".join(",s." + field for field in optional)
        con.execute("""CREATE TABLE events AS SELECT
            uid,vid,event_day,moment_ms,is_random,click,hate,
            long_view,tags,previous_video,previous_click,
            previous_hate,previous_moment,previous_tags,next_tags""" + extras + """
            FROM source_db.observed_sequences s
            WHERE s.uid IN (SELECT DISTINCT uid FROM features)
              AND s.event_day<=20220430""")
        con.execute("DETACH source_db")
        counts = {name: con.execute("SELECT COUNT(*) FROM " + name).fetchone()[0]
                  for name in ("features", "events")}
    except Exception:
        con.close()
        dst.unlink(missing_ok=True)
        raise
    con.close()
    return {"path": str(dst), "reused": False, "rows": counts}


class DevelopmentTools:
    """Read-only DuckDB plus ephemeral derived features, never source-file access."""
    def __init__(self, path: str):
        import duckdb
        self.connection = duckdb.connect(
            path, read_only=True, config={
                "enable_external_access": "false", "threads": "2", "memory_limit": "2GB"})
        self.history = []
        self.features = {"features": {}, "events": {}}
        self.feature_records = {}

    def schema(self):
        tables = ("features", "events") + tuple(
            "research_" + table for table, items in self.features.items() if items)
        return {table: [{"column": r[1], "type": r[2]}
                        for r in self.connection.execute(
                            "PRAGMA table_info('" + table + "')").fetchall()]
                for table in tables}

    def sql(self, query: str):
        clean = query.strip().rstrip(";")
        if (not clean or ";" in clean or len(clean) > 4000
                or not re.match(r"^(select|with)\b", clean, re.I)
                or FORBIDDEN.search(clean)):
            raise ValueError("Only one bounded SELECT/WITH on development data is supported")
        if any(item["sql"] == clean for item in self.history):
            raise ValueError("This exact development query already ran; change the measurement or research direction")
        executed = self.rewrite_sources(clean)
        result = self.connection.execute(
            "SELECT * FROM (" + executed + ") AS research_result LIMIT 30")
        cols = [d[0] for d in result.description]
        rows = [dict(zip(cols, (str(v) if v is not None and
                                not isinstance(v, (str, int, float, bool)) else v
                                for v in record)))
                for record in result.fetchall()]
        evidence = {"sql": clean, "columns": cols, "rows": rows}
        if executed != clean:
            evidence["executed_sql"] = executed
        self.history.append(evidence)
        return evidence

    def active_table(self, table: str) -> str:
        if table not in self.features:
            raise ValueError("Unknown development table")
        return "research_" + table if self.features[table] else table

    def rewrite_sources(self, sql: str) -> str:
        """Use the same registered views for SQL and statistical experiments."""
        for base, fields in self.features.items():
            if fields:
                sql = re.sub(r"\b(FROM|JOIN)\s+" + base + r"\b",
                             lambda m: m.group(1) + " research_" + base,
                             sql, flags=re.I)
        return sql

    def define_feature(self, name: str, expression: str, source_table: str = "features"):
        if source_table not in ("features", "events"):
            raise ValueError("Derived feature source_table must be features or events")
        if not IDENTIFIER.fullmatch(name) or name in [c["column"] for c in self.schema()[source_table]]:
            raise ValueError("Feature name must be a fresh lowercase SQL identifier")
        if not expression.strip() or len(expression) > 1000 or ";" in expression or FORBIDDEN.search(expression):
            raise ValueError("Feature expression must be an SQL scalar expression")
        if re.search(r"\b(click|hate|long_view|next_click|next_tags)\b", expression, re.I):
            raise ValueError("No future or current outcome may define a treatment; "
                             "use pre-event previous_click/previous_hate and "
                             "measure current or future outcomes separately")
        # Execute the definition on the chosen source before registering a view.
        evidence = self.sql(
            "SELECT (" + expression + ") AS " + name + ",count(*) n FROM " +
            source_table + " GROUP BY 1 ORDER BY n DESC LIMIT 8")
        levels = self.connection.execute(
            "SELECT COUNT(*) FROM (SELECT (" + expression + ") AS arm FROM " +
            self.active_table(source_table) + " GROUP BY 1 LIMIT 9)").fetchone()[0]
        if not 2 <= levels <= 8:
            raise ValueError("Experimental derived group requires 2-8 observed levels, "
                             "found " + str(levels) + "; define an actual binary"
                             " item/content or pre-event condition")
        self.features[source_table][name] = expression
        projections = ", ".join("(" + value + ") AS " + key
                                for key, value in self.features[source_table].items())
        view = "research_" + source_table
        self.connection.execute(
            "CREATE OR REPLACE TEMP VIEW " + view + " AS SELECT *, " +
            projections + " FROM main." + source_table)
        # The query gateway transparently maps the original source table
        # to this derived view when the model runs its next SELECT.
        record = {"name": name, "expression": expression, "evidence": evidence,
                  "usable_table": source_table, "explicit_view": view,
                  "source_table": source_table}
        self.feature_records[name] = record
        return record

    def feature_cards(self):
        return [{"name": item["name"], "source_table": item["source_table"],
                 "expression": item["expression"],
                 "sample_distribution": item["evidence"]["rows"][:8]}
                for item in self.feature_records.values()]

    def inspect_trajectory(self, query: str):
        # Agent may select a development user via a SELECT query, not assume an ID.
        selector = self.sql(query or
            "SELECT uid,COUNT(*) AS n FROM events GROUP BY uid ORDER BY n DESC LIMIT 1")
        if not selector["rows"] or "uid" not in selector["rows"][0]:
            raise ValueError("Trajectory selector must return a column named uid")
        uid = int(selector["rows"][0]["uid"])
        events = self.sql(
            "SELECT uid,event_day,moment_ms,vid,tags,is_random,click,hate,long_view,"
            "previous_click,previous_hate FROM events WHERE uid=" + str(uid) +
            " ORDER BY event_day,moment_ms,vid LIMIT 20")
        return {"selected_uid": uid, "selection": selector,
                "trajectory": events, "note": "chronological development events only"}

    def grounding_rows(self):
        """Observed values rather than semantic guesses about opaque tag codes."""
        available = {item["column"] for item in self.schema()["events"]}
        fields = [field for field in (
            "tags", "previous_click", "previous_hate", "is_random", "click", "hate")
            if field in available]
        rows = []
        for mode in (0, 1):
            rows.extend(self.connection.execute(
                "SELECT " + ",".join(fields) + " FROM events WHERE is_random=? "
                "ORDER BY hash(uid,vid,event_day) LIMIT 8", [mode]).fetchall())
        return [dict(zip(fields, (v if isinstance(v, (str,int,float,bool)) or v is None
                                  else str(v) for v in row))) for row in rows]

    def close(self):
        self.connection.close()


def compact_query(record: dict):
    return {"sql": record["sql"][:500],
            "executed_sql": record.get("executed_sql", record["sql"])[:500],
            "columns": record["columns"][:15],
            "rows": record["rows"][:8],
            "returned_rows": len(record["rows"]),
            "model_excerpt_only": len(record["rows"]) > 8}


def feasible_countertest_filters(tools, spec: dict) -> list[dict]:
    """Candidate subgroup comparisons supported by real paired development users.

    This returns context for the Designer, never a platform mechanism label.
    The model chooses the rival-discriminating subgroup and Workbench runs it.
    """
    if spec["source_query"].strip() or spec["unit_expression"].strip() != "uid":
        return []
    from rsi_researcher.research.discovery.workbench import _expr
    table = spec["source_table"]
    columns = {item["column"] for item in tools.schema()[table]}
    group = _expr(spec["group_expression"])
    condition = _expr(spec["population_filter"]) if spec["population_filter"].strip() else "TRUE"
    result = []
    for column in ("previous_click", "previous_hate", "is_random"):
        if column not in columns or column == spec["group_expression"]:
            continue
        sql = (f"SELECT uid,CAST(({group}) AS VARCHAR) grp,{column} val "
               f"FROM {tools.active_table(table)} "
               f"WHERE ({condition}) AND {column} IS NOT NULL GROUP BY 1,2,3")
        data = tools.connection.execute(sql).fetchall()
        groups = {}
        for uid, arm, value in data:
            groups.setdefault(value, {}).setdefault(arm, set()).add(uid)
        if len(groups) < 2:
            continue
        for value, arms in groups.items():
            paired = arms.get(spec["control_value"], set()) & arms.get(
                spec["treatment_value"], set())
            if len(paired) >= 10:
                result.append({"population_filter": (
                    f"({condition}) AND {column} = {int(value)}"),
                    "factor": column, "paired_users": len(paired)})
    return sorted(result, key=lambda item: -item["paired_users"])[:8]


def model_statistics(stats: dict) -> dict:
    """Keep every numeric conclusion but only the most supported strata in prompts.

    Complete stratum results remain in the machine-readable journal.
    """
    view = {k: v for k, v in stats.items() if k != "strata"}
    strata = stats.get("strata", {})
    view["strata_total"] = len(strata)
    view["strata"] = dict(sorted(
        strata.items(),
        key=lambda pair: min(pair[1]["n_control"], pair[1]["n_treatment"]),
        reverse=True)[:10])
    return view


def compact_step(entry: dict):
    action = entry["action"]
    result = {"round": entry["round"], "action": action["action"],
              "reason": action["reason"][:200], "concept_id": action["concept_id"],
              "concept": action["concept"][:170]}
    for k in ("result", "test", "critic_test", "critic_sql_countertest"):
        if k in entry:
            result[k] = compact_query(entry[k])
    if "trajectory" in entry:
        result["trajectory"] = compact_query(entry["trajectory"]["trajectory"])
    if "experiment" in entry:
        record = entry["experiment"]
        result["experiment"] = {
            "question": record["specification"]["question"],
            "specification": record["specification"],
            "result": model_statistics(record["result"]),
            "negative_control": (model_statistics(record["negative_control"]["result"])
                                 if record["negative_control"] else None),
        }
    if "critic_experiment" in entry:
        result["critic_experiment"] = entry["critic_experiment"]["result"]
    if "experiment_specification" in entry and "experiment" not in entry:
        result["failed_experiment"] = {
            "group": entry["experiment_specification"]["group_expression"],
            "outcome": entry["experiment_specification"]["outcome_expression"]}
    if "derived_feature" in entry:
        feature = entry["derived_feature"]
        result["derived_feature"] = {
            "name": feature["name"], "source_table": feature["source_table"],
            "expression": feature["expression"],
            "observed_values": compact_query(feature["evidence"])}
    for k in ("critique", "tool_error", "gate", "revised_concept", "literature_summary",
              "formed_concept_id", "concept_formation", "reflection",
              "contrast_audit", "reflection_veto"):
        if k in entry:
            result[k] = entry[k]
    return result


def search_openalex(queries: list[str], per_query: int = 4):
    papers, errors = [], []
    executed_queries = []
    for term in queries[:3]:
        term = term[:200]
        if not re.search(r"recommender|recommendation|personaliz|ranking", term, re.I):
            term = "recommender systems " + term
        executed_queries.append(term)
        params = urllib.parse.urlencode({
            "search": term, "per_page": per_query,
            "select": "id,doi,display_name,publication_year,abstract_inverted_index"})
        req = urllib.request.Request(
            "https://api.openalex.org/works?" + params,
            headers={"User-Agent": "RSI-Discovery/0.2 (academic research)"})
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                payload = json.load(response)
            for paper in payload.get("results", []):
                inverted = paper.get("abstract_inverted_index") or {}
                tokens = sorted((n, word) for word, positions in inverted.items()
                                for n in positions)
                papers.append({"id": paper.get("id"), "doi": paper.get("doi"),
                               "title": paper.get("display_name"),
                               "year": paper.get("publication_year"),
                               "abstract_excerpt": " ".join(w for _, w in tokens)[:1200]})
        except Exception as exc:
            errors.append({"query": term, "error": str(exc)[:200]})
    unique = {paper["id"]: paper for paper in papers if paper.get("id")}
    candidates = list(unique.values())
    domain = re.compile(
        r"recommender|recommendation|personaliz\w*|collaborative filtering|"
        r"session.based recommendation|sequential recommendation|recommendation ranking",
        re.I)
    theory = re.compile(
        r"feedback loops?.*(bias|automated decision)|"
        r"(bias|automated decision).*feedback loops?", re.I)
    relevant, rejected = [], []
    for paper in candidates:
        corpus = paper["title"] + " " + paper["abstract_excerpt"]
        if domain.search(corpus) or theory.search(paper["title"]):
            relevant.append(paper)
        else:
            rejected.append({"id": paper["id"], "title": paper["title"]})
    return {"queries": queries, "executed_queries": executed_queries,
            "papers": relevant[:12], "excluded_irrelevant": rejected,
            "errors": errors, "retrieval_complete": not errors}


def literature_review(model, concept: dict, episode: str):
    agent = ResearchAgent(model)
    agent.role = "literature-researcher"
    agent.instruction = (
        "Form 2-3 English searches for MECHANISM-equivalent work on recommendation, "
        "feedback loops, algorithmic bias, behavior sequences and user preferences. "
        "Search by mechanism and measurement, not only a named Bias.")
    queries = agent.act(concept, episode + "-search", QUERY_SCHEMA)["search_queries"]
    retrieved = search_openalex(queries)
    agent.instruction = (
        "Review paper abstracts and titles relative to this EXACT mechanism. "
        "Cite only returned paper IDs. Distinguish prior work overlap from the specific "
        "unresolved contribution. Retrieval is NOT exhaustive proof of novelty. "
        "Missing evidence means insufficient_literature.")
    report = agent.act({"concept": concept, "papers": retrieved}, episode + "-review", NOVELTY_SCHEMA)
    allowed = {p["id"] for p in retrieved["papers"]}
    if not set(report["strongest_prior_work_ids"]).issubset(allowed):
        raise ValueError("Literature reviewer cited nonexistent paper IDs")
    if retrieved["errors"] or len(retrieved["papers"]) < 2:
        report["verdict"] = "insufficient_literature"
    return {"retrieval": retrieved, "assessment": report}


def _concept_summary(item: dict):
    return {"id": item["id"], "version": item["version"], "status": item["status"],
            "concept": item["claim"]["concept"], "mechanism": item["claim"]["mechanism"],
            "rival": item["claim"]["rival"], "measurement": item["claim"]["measurement"],
            "falsifier": item["claim"]["falsifier"],
            "development_tests": len(item["tests"]),
            "literature_verdict": (
                item["literature"]["assessment"]["verdict"] if item["literature"] else "not_checked"),
            "critic_decision": (item["critique"]["decision"] if item["critique"] else "not_reviewed")}


def compact_opportunity_pool(pool) -> dict:
    """Research state for 8B models: keep decisive evidence, not repeated transcripts."""
    return {"focus": pool.focus, "opportunities": [{
        "id": item["id"], "status": item["status"],
        "phenomenon": item["phenomenon"][:180],
        "mechanism": item["mechanism"][:280],
        "rival": item["rival"][:250],
        "falsifier": item["falsifier"][:220],
        "scientific_contract": {
            k: (v[:220] if isinstance(v, str) else v)
            for k, v in item.get("scientific_contract", {}).items()},
        "bias_reference": {
            k: v for k, v in item.get("bias_relevance", {}).items()
            if k in ("reference_kind", "target_reference",
                     "reference_identification", "reference_measured",
                     "numeric_discrepancy", "reference_scope")},
        "selected_contrast": {
            k: v for k, v in item.get("selection_contract", {}).items()
            if k in ("source_table", "group_expression", "outcome_expression",
                     "control_value", "treatment_value", "analysis",
                     "prediction_if_mechanism", "prediction_if_rival")},
        "latest_observation": [{
            "sql": e.get("sql", "")[:130],
            "rows": e.get("rows", [])[:2]}
            for e in item.get("observed_evidence", [])[-1:]],
        "latest_test": [{
            k: v for k, v in t.items()
            if k in ("initial_estimate", "initial_ci", "initial_users",
                     "counter_estimate", "counter_ci", "counter_users",
                     "critic_reason")}
            for t in item.get("tests", [])[-1:]],
        "scientific_evidence_gaps": item.get("scientific_evidence_gaps", []),
        "next_research_decision": item.get("next_research_decision"),
    } for item in list(pool.records.values())[-5:]]}


def compact_observation_cards(cards: list[dict]) -> list[dict]:
    return [{"observation_id": c["observation_id"], "question": c["question"],
             "measurement_sql": c["measurement_sql"][:260],
             "measured_rows": c["measured_rows"][:5],
             "caveat": c["caveat"]}
            for c in cards]


def _save(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def scientific_evidence_gaps(concept: dict, opportunity: dict) -> list[str]:
    """Evidence gaps based on executed artifacts, not prose or model confidence."""
    gaps = []
    bias = opportunity.get("bias_relevance", {})
    reference = bias.get("numeric_discrepancy", {})
    if not reference.get("measured"):
        gaps.append("numeric_reference_deviation_unmeasured")
    elif not reference.get("normative_target_validated"):
        gaps.append("normative_reference_not_identified")
    if not any(t.get("requested_by") == "critic" and "workbench" in t
               for t in concept.get("tests", [])):
        gaps.append("no_executed_inferential_rival_countertest")
    critique = concept.get("critique") or {}
    if critique and not critique.get("observable_valid"):
        gaps.append("experiment_observable_invalid")
    if critique and not critique.get("rival_discriminated"):
        gaps.append("competing_mechanism_not_discriminated")
    # Consequence cannot be inferred from an exposure-mode share difference.
    gaps.append("recommendation_consequence_not_identified")
    return gaps


def research_funnel(concepts: list[dict], pool) -> dict:
    "Evidence-stage counts, never free-text 'successful discovery' claims."
    workbench = sum(any("workbench" in trial for trial in c["tests"]) for c in concepts)
    discriminated = sum(
        bool((c.get("critique") or {}).get("rival_discriminated"))
        and any("workbench" in t and t.get("requested_by") == "critic"
                for t in c["tests"])
        for c in concepts)
    measured = sum(bool(c["bias_relevance"].get(
        "numeric_discrepancy", {}).get("measured"))
        for c in pool.records.values())
    return {
        "candidates_proposed": len(concepts),
        "candidates_with_executed_workbench": workbench,
        "candidates_with_rival_discrimination": discriminated,
        "candidates_with_numeric_reference_diagnostic": measured,
        "candidates_with_credible_distortion_measurement": sum(
            bool(r["bias_relevance"].get("numeric_discrepancy", {}).get(
                "normative_target_validated"))
            for r in pool.records.values()),
        "workbench_completion_rate": workbench / len(concepts) if concepts else 0.0,
        "independently_confirmed": 0,
    }


def research_evidence_signature(tools, concepts: list[dict], pool) -> tuple:
    """Evidence changes, not elapsed turns or mere renamed hypotheses."""
    return (len(tools.history), len(tools.feature_records),
            sum(len(c["tests"]) + len(c["revisions"]) for c in concepts),
            sum(len(p.get("tests", [])) for p in pool.records.values()),
            sum(bool(p.get("bias_relevance", {}).get("numeric_discrepancy", {}).get(
                "measured")) for p in pool.records.values()))


def run_open_discovery(source_db: str, output: str, model_config: dict,
                       max_rounds: int | None = None, min_queries: int = 2,
                       development_path: str | None = None) -> dict:
    """Continue until a research decision or information stall; never read holdout.

    Explicit max_rounds retains the legacy bounded mode for evaluation.
    """
    if (max_rounds is not None and
            (type(max_rounds) is not int or max_rounds < 3 or max_rounds <= min_queries)):
        raise ValueError("max_rounds must exceed min_queries and be >= 3")
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=False)
    dev_path = development_path or str(dest / "development_only.duckdb")
    prep = development_db(source_db, dev_path)
    _save(dest / "development_manifest.json", {
        "development": prep, "holdout_exposed": False,
        "source_fingerprint_input": str(Path(source_db).resolve())})
    tools = DevelopmentTools(dev_path)
    # Development-only grounding, cached once. Tag values are opaque IDs, not words.
    data_examples = tools.grounding_rows()
    from rsi_researcher.research.discovery.observations import build_observation_cards
    discovery_options = model_config.get("discovery", {})
    observation_cards = (build_observation_cards(tools)
                         if discovery_options.get("observation_cards", True) else [])
    _save(dest / "observation_cards.json", observation_cards)
    import hashlib
    _save(dest / "discovery_protocol.json", {
        "observation_cards_enabled": discovery_options.get("observation_cards", True),
        "compact_experiment_enabled": discovery_options.get("compact_experiment", True),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_config": model_config.get("llm", {}),
        "research_stop_policy": ("evidence_or_stall" if max_rounds is None
                                 else "fixed_rounds"),
        "max_rounds": max_rounds,
        "holdout_exposed": False})
    from rsi_researcher.research.discovery.workbench import (
        ExperimentalWorkbench, EXPERIMENT_SCHEMA)
    workbench = ExperimentalWorkbench(tools)
    model_dir = dest / "model_calls"
    model_dir.mkdir()
    # Live science mode has no model-request, token or wall-clock research quota.
    # Each HTTP call still has its ordinary timeout and retry policy. Bounded
    # quotas remain for controlled comparisons when --max-rounds is explicit.
    limits = (RunLimits(model_dir, max_requests=5 * max_rounds + 4,
                        max_tokens=250000, max_seconds=2400)
              if max_rounds is not None else None)
    model = create_model(model_dir,
                         limits.max_requests if limits else float("inf"),
                         limits, model_config)
    scientist = ResearchAgent(model)
    scientist.role = "open-discovery-scientist"
    scientist.instruction = (
        "You are an autonomous recommender-systems research scientist, not a "
        "candidate classifier. Investigate user and recommendation SEQUENCES. "
        "Our goal is a CORRECTABLE SYSTEMATIC DISTORTION relative to a "
        "credible target reference, not just ordinary behavior correlation. "
        "A behavioral pattern is not automatically a recommender Bias. "
        "Generic popularity, tag frequency, feedback loop, and historical "
        "CTR correlations are NOT new Bias discoveries. Start with numbered "
        "measured observation cards; seek a distinctive CONDITIONAL "
        "distortion in ITEM allocation against an explicit supported "
        "reference, not a mere cross-policy CTR contrast. "
        "Choose the next ACTION yourself. Use SQL, inspect an actual trajectory, "
        "define a NEW derived feature, form multiple competing concepts, request "
        "an EXPERIMENT action so a Designer can specify test/control, outcome, "        " user-level inference, placebo and falsifier; revise based on results, "
        "and consult literature. Existing columns are observational conveniences "
        "NOT predefined Bias categories. SQL uses exactly the given column names; "
        "never invent feature1 or generic names. Use CASE WHEN bins rather than "
        "grouping by unbinned continuous timestamps. Do not join two event-level "
        "tables on uid alone. If a proposal is contradicted, revise or discard it. "
        "Tags are OPAQUE numeric ID strings (such as '39,43'), NOT natural-language "
        "topics. 'hate' is a binary RESPONSE column, not a tag name; do not query "
        "tags LIKE '%hate%'. Current click/hate/long_view are OUTCOMES and cannot "
        "be used to define the independent treatment variable. "
        "When a numerical tool says a comparison arm has zero observations, inspect "
        "actual values and choose a supported contrast, never submit it unchanged. "
        "A descriptive difference is NOT evidence of an algorithmically harmful Bias. "
        "Do not claim independent replication or intervention from this dev dataset. "
        "For define_feature choose source_table events to access previous_tags or next_tags; "
        "features has no such columns. Derived features using current click in the "
        "definition cannot be used as an independent predictor of current click. "
        "When a concept list is available use EXACT H001/H002 concept_id; never "
        "invent concept_id strings. Use test then revise if evidence changes the claim. "
        "For any action, fill the JSON fields; irrelevant strings can be empty. "
        "A concept is NOT frozen simply because it has an interesting name.")
    critic = ResearchAgent(model)
    critic.role = "discovery-critic"
    critic.instruction = (
        "You are an independent scientific critic. Evaluate this development "
        "hypothesis, its actual measurements and strongest alternative. "
        "Suggest one concrete SQL test on existing development tables whenever "
        "the rival mechanism is not discriminated. If the evidence remains "
        "descriptive, NEVER claim causal identification. Mark ready only when "
        "the observable really measures the proposed relationship, the rival "
        "has a relevant development falsification check, and obvious logging "
        "artifacts were probed. A test can justify provisional freezing for "
        "independent confirmation, not a causal Bias claim.")
    scientist.instruction += BIAS_RESEARCH_CONTRACT
    critic.instruction += (
        BIAS_RESEARCH_CONTRACT
        + " Audit distortion validity, reference validity, H1/H2 "
        "mechanism discrimination, consequence, and current evidence stage "
        "separately. A Critic assertion of ready cannot replace a measured "
        "reference gap or independently executed countertest. "
        "Report the specific identification or test failure in reason."
    )
    history, concepts = [], []
    experiment_failures = {}
    opportunity_pool = ResearchOpportunityPool()
    last = ""
    stop_reason = ""
    no_information_actions = 0
    try:
        for step in (itertools.count() if max_rounds is None
                     else range(max_rounds)):
            evidence_before = research_evidence_signature(tools, concepts, opportunity_pool)
            if (step >= 3 and tools.history and
                    (not concepts or (
                        (max_rounds is None or max_rounds - step >= 3)
                        and all(c["status"] in ("rejected_development", "design_blocked")
                                for c in concepts)))):
                # Return to candidate formation after *all* prior hypotheses
                # were rejected, instead of repeatedly testing dead ends.
                # A dedicated concept-formation turn prevents tool-format repairs
                # from indefinitely displacing scientific hypothesis creation.
                # It does not predefine variables, categories or candidate cards.
                former = ResearchAgent(model)
                former.role = "concept-formation"
                former.instruction = (
                    "From ONLY the actual development observations, formulate "
                    "TWO different testable scientific concepts, each with a "
                    "mechanism, substantive competing explanation, falsifying "
                    "prediction, new measurement and executable test SQL. "
                    "Propose conditional, falsifiable relationships based on "
                    "observation IDs, not names of familiar Biases. State "
                    "which item allocation, reference and H1/H2 predictions "
                    "the same numerical test could discriminate. "
                    "For each concept specify an observable prediction if the mechanism "
                    "is true and explain what relevance, logging or feedback reference "
                    "would reveal a SYSTEMATIC DISTORTION. Do not invent target values. "
                    "is true, a DIFFERENT observable prediction if its rival is true, "
                    "and what follow-up observation or experiment best separates them. "
                    "Use the actual measured rows; do not make up evidence, effect "
                    "sizes or sample counts. We want research opportunity selection, "
                    "not generic significant findings. "
                    "Do not claim causality or novelty. Use exact schema columns.")
                former.instruction += (
                    BIAS_RESEARCH_CONTRACT
                    + " Begin each concept with the suspected recommender "
                    "decision, affected users/items, outcome-reference "
                    "discrepancy, and distinct rival predictions BEFORE SQL. "
                    "Declare reference_status and next_information_needed. "
                    "Unknown reference is a valid lead but not established bias."
                )
                concept_contract = CONCEPT_PAIR_SCHEMA
                if model_config.get("llm"):
                    item = CONCEPT_PAIR_SCHEMA["properties"]["concepts"]["items"]
                    required = list(item["required"]) + [
                        "target_reference", "reference_identification",
                        "distortion_quantity", "distortion_estimate",
                        "prediction_if_mechanism", "prediction_if_rival",
                        "affected_population", "mechanism_entry_point",
                        "reference_status", "next_information_needed",
                        "discriminating_experiment"]
                    concept_contract = {**CONCEPT_PAIR_SCHEMA, "properties": {
                        "concepts": {**CONCEPT_PAIR_SCHEMA["properties"]["concepts"],
                                     "items": {**item, "required": required}}}}
                formed = former.act({
                    "observations": [compact_query(x) for x in tools.history[-4:]],
                    "observation_cards": observation_cards,
                    "tables": tools.schema(),
                    "existing_derived_features": tools.features,
                    "rejected_prior_mechanisms": [
                        _concept_summary(c) for c in concepts if
                        c["status"] == "rejected_development"],
                    "next_question_instruction": "Propose a mechanism that makes "
                        "a NEW falsifiable prediction instead of renaming a "
                        "previously abandoned hypothesis.",
                }, "concept-formation-" + str(step), concept_contract)
                stage_ids = []
                for hypothesis in formed["concepts"]:
                    identifier = "H" + str(len(concepts) + 1).zfill(3)
                    claim = {**hypothesis, "concept_id": identifier,
                             "query": hypothesis["suggested_sql"],
                             "action": "form_concept",
                             "reason": "scientific concept formation from measured evidence",
                             "feature_name": "", "expression": "",
                             "source_table": "features"}
                    concepts.append({
                        "id": identifier, "version": 1, "status": "investigating",
                        "claim": claim, "tests": [], "revisions": [],
                        "literature": None, "critique": None})
                    opportunity_pool.register(identifier, hypothesis, tools.history)
                    stage_ids.append(identifier)
                stage_action = {"action": "concept_formation", "reason": "hypothesis synthesis",
                                "concept_id": "", "concept": "", "query": ""}
                history.append({"round": step, "action": stage_action,
                                "concept_formation": stage_ids})
                _save(dest / "concept_formation.json", formed)
                # The autonomous choice runs INSIDE the same research cycle and
                # shares the same logged Qwen/tool calls. Fixtures without an
                # actual configured LLM still exercise the normal action route.
                if model_config.get("llm"):
                    chooser = ResearchAgent(model)
                    chooser.role = "research-opportunity-selector"
                    chooser.instruction = (
                        "Choose which AGENT-PROPOSED phenomenon to investigate next. "
                        "Compare the mechanism-vs-rival predictions; select the "
                        "observation or statistical experiment where they disagree "
                        "most, using actual executable support and information gain. "
                        "State distinct observable predictions for BOTH the proposed "
                        "mechanism and its rival; repeating the same prediction is "
                        "not an informative experiment. "
                        "Choose one concrete comparison that separates H1/H2: "
                        "real group and arm labels, one NUMERIC per-event outcome "
                        "(not tags/next_tags or AVG), optional development cohort "
                        "filter, and two DIFFERENT expected directions on that "
                        "SAME contrast. Preserve the Agent-created registered "
                        "variable name where appropriate. Your predicted directions "
                        "are hypotheses, NOT observed results. Prefer at least 20 "
                        "independent paired users, and never confuse observations "
                        "with interventions. Do NOT rank by p-value or invent an "
                        "additional candidate. Specify which credible reference "
                        "distribution or relevance signal could expose the distortion, "
                        "the causal link in collection/learning/ranking, and a corrective "
                        "intervention. If no reference is currently measurable, use "
                        "reference_kind=none and do not call it a Bias. "
                        "An observe decision needs executable SQL.")
                    chooser.instruction += (
                        BIAS_RESEARCH_CONTRACT
                        + " Prefer the next observation that resolves the "
                        "reference identification gap or separates H1/H2. "
                        "Never mistake a statistical treatment/control group "
                        "for an actual intervention on the recommender."
                    )
                    available_records = list(tools.feature_records.values())
                    # The first discriminating experiment must use a variable
                    # previously created and measured by the scientist.
                    allowed_tables = {record["source_table"] for record in available_records}
                    allowed_groups = [record["name"] for record in available_records]
                    # Ground native comparison dimensions in measured rows.
                    # They are possible measurements, never pre-named Bias cards.
                    native_levels = {}
                    event_columns = {c["column"] for c in tools.schema()["events"]}
                    for card in observation_cards:
                        for row in card["measured_rows"]:
                            for name in ("tab", "previous_click", "previous_hate"):
                                if name in event_columns and row.get(name) is not None:
                                    native_levels.setdefault(name, set()).add(str(row[name]))
                    native_levels = {
                        name: levels for name, levels in native_levels.items()
                        if 2 <= len(levels) <= 8}
                    if native_levels:
                        allowed_tables.add("events")
                        allowed_groups.extend(native_levels)
                    numeric_prefixes = ("BOOLEAN", "INTEGER", "INT", "BIGINT",
                                        "SMALLINT", "TINYINT", "HUGEINT",
                                        "DOUBLE", "FLOAT", "REAL", "DECIMAL",
                                        "UBIGINT", "USMALLINT", "UTINYINT")
                    observed_labels = sorted(({
                        str(row[record["name"]])
                        for record in available_records
                        for row in record["evidence"]["rows"]
                        if record["name"] in row and row[record["name"]] is not None
                    } | {v for levels in native_levels.values() for v in levels}))
                    actual_numeric_columns = sorted({
                        col["column"] for table in allowed_tables
                        for col in tools.schema()[table]
                        if col["type"].upper().startswith(numeric_prefixes)})
                    constrained_schema = OPPORTUNITY_CHOICE_SCHEMA
                    if allowed_groups and actual_numeric_columns:
                        constrained_schema = {
                            **OPPORTUNITY_CHOICE_SCHEMA, "properties": {
                                **OPPORTUNITY_CHOICE_SCHEMA["properties"],
                                "contrast_source": {"type": "string",
                                                    "enum": sorted(allowed_tables)},
                                "contrast_group": {"type": "string",
                                                   "enum": allowed_groups},
                                "contrast_outcome": {"type": "string",
                                                     "enum": actual_numeric_columns},
                                "control_value": {"type": "string", "enum": observed_labels},
                                "treatment_value": {"type": "string", "enum": observed_labels},
                            }}
                    choice = chooser.act({
                        "available_numeric_outcomes_by_source": {
                            table: [c["column"] for c in tools.schema()[table]
                                    if c["column"] in ("click", "long_view",
                                                       "hate", "recent_click")]
                            for table in allowed_tables},
                        "already_registered_groups": [
                            r["name"] for r in available_records],
                        "native_observed_groups": {
                            col: sorted(levels) for col, levels in native_levels.items()},
                        "scientific_restriction": (
                            "Exposure-mode vs click difference alone cannot "
                            "discriminate platform bias from user and item mix. "
                            "Use a measured contextual interaction or an "
                            "Agent-defined item-side binary feature; if no "
                            "valid contrast exists, gather evidence instead."),
                        "opportunities": compact_opportunity_pool(opportunity_pool),
                        "recent_observations": [
                            {"sql": q["sql"][:160], "rows": q["rows"][:3]}
                            for q in tools.history[-2:]],
                        "observation_cards": compact_observation_cards(observation_cards),
                        "available_schema": {
                            table: [c["column"] for c in cols]
                            for table, cols in tools.schema().items()},
                        "unresolved_research_question":
                            "Find the next credible reference or H1/H2 contrast;"
                            " no predetermined research-round quota is imposed.",
                    }, "opportunity-choice-" + str(step),
                        constrained_schema)
                    def check_choice(plan):
                        opportunity_pool.check_prediction(plan)
                        if (plan["contrast_group"] == "is_random"
                                and plan["contrast_outcome"] in ("click", "long_view", "hate")):
                            raise ValueError(
                                "Raw exposure-mode response gaps cannot distinguish "
                                "ranking distortion from differences in offered items. "
                                "Choose an item-side allocation or conditioned reference "
                                "comparison with different H1/H2 predictions.")
                        if plan["reference_kind"] == "randomized_exposure":
                            record = tools.feature_records.get(plan["contrast_group"])
                            expression = record["expression"] if record else plan["contrast_group"]
                            if ("tags" not in expression and "vid" not in expression
                                    and "author" not in expression):
                                raise ValueError("Randomized exposure reference needs "
                                                 "an actual ITEM feature, not is_random "
                                                 "or a historical user-state contrast")
                            if "is_random" in expression or "click" in expression:
                                raise ValueError("Cannot use exposure mode or outcome "
                                                 "to define the item-side treatment")
                        preview = {
                            "question": plan["discriminating_prediction"],
                            "source_table": plan["contrast_source"],
                            "source_query": "",
                            "population_filter": plan["contrast_filter"],
                            "unit_expression": "uid",
                            "group_expression": plan["contrast_group"],
                            "control_value": plan["control_value"],
                            "treatment_value": plan["treatment_value"],
                            "outcome_expression": plan["contrast_outcome"],
                            "strata_expression": "",
                            "analysis": plan["analysis"],
                            "placebo_source_query": "",
                            "placebo_group_expression": "",
                            "prediction": plan["prediction_if_mechanism"],
                            "falsifier": plan["prediction_if_rival"],
                            "meaningful_margin": 0.0}
                        support = workbench.inspect_plan(preview)
                        if not support["valid_support"]:
                            raise ValueError("Chosen comparison lacks independent user support: "
                                             + str(support))
                        return support

                    try:
                        support = check_choice(choice)
                    except Exception as error:
                        # A condition can accidentally define the treatment
                        # itself. Supply evidence about the unfiltered comparison
                        # so Qwen can choose a scientifically meaningful repair.
                        unfiltered_support = None
                        try:
                            unfiltered = {**choice, "contrast_filter": ""}
                            unfiltered_support = check_choice(unfiltered)
                        except Exception as unfiltered_error:
                            unfiltered_support = {"reason": str(unfiltered_error)[:220]}
                        try:
                            repair_schema = constrained_schema
                            if unfiltered_support.get("valid_support"):
                                # The only support-preserving relaxation actually
                                # measured here. No hand-coded bias-specific
                                # candidate feature or experimental policy.
                                repair_schema = {**constrained_schema, "properties": {
                                    **constrained_schema["properties"],
                                    "contrast_filter": {"type": "string", "enum": [""]}}}
                            choice = chooser.act({
                                "rejected_choice": choice,
                                "numeric_or_support_error": str(error),
                                "same_contrast_without_population_filter": unfiltered_support,
                                "registered_variable_cards": tools.feature_cards(),
                                "available_numeric_outcomes": actual_numeric_columns,
                                "instruction": "This subgroup excludes a comparison "
                                               "arm. Select an ACTUALLY supported "
                                               "population; an unfiltered contrast "
                                               "can be tested only if the mechanism "
                                               "and rival still meaningfully differ. "
                                               "Otherwise choose a different mechanism "
                                               "in the next research turn.",
                            }, "opportunity-choice-repair-" + str(step),
                                repair_schema)
                            support = check_choice(choice)
                        except Exception as repair_error:
                            support = {"valid_support": False,
                                       "reason": str(repair_error)[:350]}
                    if not support["valid_support"]:
                        history.append({
                            "round": step, "action": {
                                "action": "opportunity_selection_failed",
                                "reason": "No supported discriminating contrast",
                                "concept_id": choice.get("selected_concept_id", ""),
                                "query": "", "concept": ""},
                            "rejected_opportunity_selection": choice,
                            "selection_feasibility": support,
                        })
                        _save(dest / "research_opportunities.json",
                              opportunity_pool.snapshot())
                        _save(dest / "journal.json", history)
                    else:
                        opportunity_pool.choose(
                            choice["selected_concept_id"], choice["reason"],
                            choice["next_sql"] if choice["next_action"] == "observe" else "")
                        opportunity_pool.record_selection_design(
                            choice["selected_concept_id"], choice, support)
                        chosen_item = opportunity_pool.records[choice["selected_concept_id"]]
                        chosen_item["prediction_if_mechanism"] = choice["prediction_if_mechanism"]
                        chosen_item["prediction_if_rival"] = choice["prediction_if_rival"]
                        chosen_item["discriminating_experiment"] = choice["discriminating_prediction"]
                        chosen_item["experiment_cost"] = choice["experimental_cost"]
                        chosen_entry = {
                            "round": step,
                            "action": {"action": "select_opportunity",
                                       "reason": choice["reason"], "concept_id":
                                       choice["selected_concept_id"], "query":
                                       choice["next_sql"], "concept": ""},
                            "opportunity_selection": choice,
                            "selection_feasibility": support}
                        if choice["next_action"] == "observe" and choice["next_sql"].strip():
                            try:
                                chosen_entry["result"] = tools.sql(choice["next_sql"])
                                opportunity_pool.observation(
                                    choice["selected_concept_id"],
                                    chosen_entry["result"]["sql"],
                                    chosen_entry["result"]["rows"])
                            except Exception as exc:
                                chosen_entry["tool_error"] = str(exc)[:300]
                        history.append(chosen_entry)
                _save(dest / "research_opportunities.json", opportunity_pool.snapshot())
                _save(dest / "journal.json", history)
            instruction = {
                "tables": tools.schema(),
                "development_value_examples": data_examples,
                "observation_cards": compact_observation_cards(observation_cards),
                "derived_features": tools.features,
                "registered_variable_cards": tools.feature_cards(),
                "concepts": [_concept_summary(c) for c in concepts[-4:]],
                "research_opportunity_pool": compact_opportunity_pool(opportunity_pool),
                "active_concept_ids": [c["id"] for c in concepts
                    if c["status"] not in ("rejected_development", "design_blocked")],
                "abandoned_concept_ids": [c["id"] for c in concepts
                    if c["status"] in ("rejected_development", "design_blocked")],
                "opportunity_selection_principle": (
                    "You own the next research choice: select_opportunity with a "
                    "concept_id and optionally a fresh SQL observation, or choose "
                    "SQL/define_feature/experiment directly. Prefer the question "
                    "whose next MEASURED outcome would distinguish competing "
                    "mechanisms. Explain the discriminating prediction, expected "
                    "information and executable support. Do NOT prioritize "
                    "a hypothesis merely because its p-value or effect is large."),
                "recent_evidence": [compact_step(s) for s in history[-2:]],
                "research_stop_policy": (
                    "continue on new evidence; stop on scientific exhaustion, "
                    "confirmed freeze, repeated no-information actions or interruption"),
                "no_information_actions": no_information_actions,
                "successful_queries": len(tools.history),
                "minimum_queries_before_freeze": min_queries,
                "available_actions": ACTION_SCHEMA["properties"]["action"]["enum"],
                "research_mission": (
                    "Find a previously unspecified systematic recommender "
                    "distortion relative to a credible reference; distinguish "
                    "it from real user taste with H1/H2 predictions, and "
                    "measure it before calling it a bias."),
                "sql_hint": "For new concepts measure conditional interactions, with counts and user units.",
                "next_action_guidance": (
                    "Use EXACT existing concept IDs. Select the next action by "
                    "the unresolved scientific gaps in the opportunity pool: "
                    "identify a defensible reference when unmeasured, then test "
                    "a contrast where the mechanism and preference rival disagree; "
                    "revise or abandon when contradicted. Use each opportunity's "
                    "next_research_decision (including reference gaps, numerical "
                    "counterevidence and execution failures), not just its name. "
                    "Request literature before freeze." if concepts else
                    "Inspect observed allocation and response first; identify "
                    "which recommender decision and reference would distinguish "
                    "a systematic distortion from preferences. Use SQL or a "
                    "trajectory if reference support is unclear; define_feature "
                    "only when a chosen scientific question needs it."
                    if not any(tools.features.values()) else
                    "Test the registered variable with a conditional comparison and "
                    "form a falsifiable hypothesis with a competing explanation."),
            }
            live_ids = instruction["active_concept_ids"]
            live_actions = ACTION_SCHEMA["properties"]["action"]["enum"]
            if not live_ids:
                live_actions = [a for a in live_actions if a not in (
                    "select_opportunity", "test", "experiment",
                    "revise", "literature", "freeze")]
            model_action_schema = ACTION_SCHEMA
            if model_config.get("llm"):
                # Repeated identical SQL is a dead-end, not another hypothesis test.
                if len(history) >= 2 and all(
                        "This exact development query already ran" in e.get(
                            "tool_error", "") for e in history[-2:]):
                    live_actions = [a for a in live_actions
                                    if a not in ("sql", "inspect_trajectory", "test")]
                    instruction["next_action_guidance"] += (
                        " The last two identical queries yielded no information. "
                        "Choose a NEW measurable variable, research concept, or "
                        "opportunity; SQL/trajectory repetition is unavailable.")
                valid_experiment = bool(tools.feature_records) or any(
                    opportunity_pool.records[ident].get("selection_contract")
                    for ident in live_ids if ident in opportunity_pool.records)
                if not valid_experiment:
                    live_actions = [a for a in live_actions if a != "experiment"]
                    instruction["next_action_guidance"] += (
                        " Experiment is not available: first choose a supported "
                        "research contrast with the opportunity selector or "
                        "define and measure an item/context feature.")
                model_action_schema = {**ACTION_SCHEMA, "properties": {
                    **ACTION_SCHEMA["properties"],
                    "action": {"type": "string", "enum": live_actions},
                    "concept_id": {"type": "string", "enum": ["", *live_ids]}}}
            decision = scientist.act(
                instruction, "open-" + str(step), model_action_schema)
            entry = {"round": step, "action": decision}
            act = decision["action"]
            selected = next((c for c in concepts if c["id"] == decision["concept_id"]), None)
            try:
                if act == "select_opportunity":
                    chosen = opportunity_pool.choose(
                        decision["concept_id"], decision["reason"], decision["query"])
                    entry["opportunity_selection"] = {
                        "id": chosen, "reason": decision["reason"],
                        "next_observation": decision["query"]}
                    if decision["query"].strip():
                        entry["result"] = tools.sql(decision["query"])
                        opportunity_pool.observation(
                            chosen, entry["result"]["sql"], entry["result"]["rows"])
                elif act == "sql":
                    entry["result"] = tools.sql(decision["query"])
                    if opportunity_pool.focus:
                        opportunity_pool.observation(
                            opportunity_pool.focus, entry["result"]["sql"],
                            entry["result"]["rows"])
                elif act == "inspect_trajectory":
                    entry["trajectory"] = tools.inspect_trajectory(decision["query"])
                elif act == "define_feature":
                    entry["derived_feature"] = tools.define_feature(
                        decision["feature_name"], decision["expression"], decision["source_table"])
                elif act == "form_concept":
                    if len(tools.history) < 1:
                        raise ValueError("Observe development data before forming a concept")
                    if not all(decision[k].strip() for k in
                               ("concept", "mechanism", "rival", "falsifier", "measurement")):
                        raise ValueError("A concept needs mechanism, rival, falsifier and measurement")
                    identifier = "H" + str(len(concepts) + 1).zfill(3)
                    selected = {"id": identifier, "version": 1, "status": "investigating",
                                "claim": decision, "tests": [], "revisions": [],
                                "literature": None, "critique": None}
                    concepts.append(selected)
                    opportunity_pool.register(identifier, {
                        **decision, "suggested_sql": decision["query"]},
                        tools.history)
                    entry["formed_concept_id"] = identifier
                elif act == "test":
                    if selected is None:
                        raise ValueError("Select an existing concept_id for an active test")
                    measured = tools.sql(decision["query"])
                    entry["test"] = measured
                    entry["contrast_audit"] = {
                        "has_two_result_groups": len(measured["rows"]) >= 2,
                        "note": "One aggregate row cannot establish a between-condition contrast"}
                    selected["tests"].append({
                        "version": selected["version"],
                        "test_question": decision["reason"],
                        "predicted_falsifier": selected["claim"]["falsifier"],
                        "result": measured})
                    critique = critic.act(
                        {"concept": _concept_summary(selected),
                         "original_claim": selected["claim"],
                         "test": compact_query(measured),
                         "recent": [compact_query(q) for q in tools.history[-3:]],
                         "tables": tools.schema()},
                        "critique-" + str(step), CRITIC_SCHEMA)
                    if not entry["contrast_audit"]["has_two_result_groups"]:
                        critique["observable_valid"] = False
                        critique["rival_discriminated"] = False
                        critique["reason"] += " [NUMERICAL AUDIT: only one comparison row; the proposed contrast was not measured.]"
                    entry["critique"] = critique
                    selected["critique"] = critique
                    if critique["decision"] == "test" and critique["next_sql"]:
                        try:
                            test = tools.sql(critique["next_sql"])
                            entry["critic_test"] = test
                            selected["tests"].append({
                                "version": selected["version"],
                                "test_question": critique["reason"],
                                "predicted_falsifier": selected["claim"]["falsifier"],
                                "result": test, "requested_by": "critic"})
                        except Exception as exc:
                            entry["critic_test_error"] = str(exc)[:270]
                    if critique["decision"] == "reject":
                        selected["status"] = "rejected_development"
                    elif critique["decision"] == "revise":
                        selected["status"] = "needs_revision"
                    elif critique["decision"] == "test" and sum(
                            t["version"] == selected["version"] for t in selected["tests"]) >= 2:
                        reflector = ResearchAgent(model)
                        reflector.role = "discovery-reflection"
                        reflector.instruction = (
                            "Review the measured development tests and Critic's repeated "
                            "failure to exclude an alternative. Choose whether to REVISE "
                            "the scientific concept/metric (using a clearly different "
                            "falsification plan), ABANDON it, or consult LITERATURE. "
                            "Do not repeat the same SQL, invent significant results, or "
                            "treat correlation as a confirmed Bias.")
                        reflector.instruction += BIAS_RESEARCH_CONTRACT
                        reflection = reflector.act({
                            "hypothesis": selected["claim"],
                            "tests": [compact_query(t["result"]) for t in selected["tests"][-3:]],
                            "critique": critique,
                        }, "reflection-" + str(step), REFLECTION_SCHEMA)
                        entry["reflection"] = reflection
                        if reflection["decision"] == "revise" and all(
                                reflection[k].strip() == selected["claim"][k].strip()
                                for k in ("concept", "mechanism", "rival", "falsifier", "measurement")):
                            entry["reflection_veto"] = (
                                "No substantive hypothesis or measurement change; "
                                "this is not a new research version")
                            selected["status"] = "needs_revision"
                        elif reflection["decision"] == "revise":
                            selected["revisions"].append({
                                "previous": selected["claim"], "reason": reflection["reason"],
                                "triggering_tests": len(selected["tests"])})
                            selected["claim"] = {
                                **selected["claim"],
                                **{key: reflection[key] for key in (
                                    "concept", "mechanism", "rival", "falsifier", "measurement")},
                                "query": reflection["next_sql"], "reason": reflection["reason"]}
                            selected["version"] += 1
                            selected["status"] = "investigating"
                            selected["critique"] = None
                            entry["revised_concept"] = _concept_summary(selected)
                        elif reflection["decision"] == "abandon":
                            selected["status"] = "rejected_development"
                        elif reflection["decision"] == "literature":
                            selected["literature"] = literature_review(
                                model, selected["claim"], "reflection-literature-" + str(step))
                            entry["literature_summary"] = selected["literature"]["assessment"]
                elif act == "experiment":
                    if selected is None:
                        raise ValueError("Select an existing concept_id for a statistical experiment")
                    if selected["status"] in ("rejected_development", "design_blocked"):
                        raise ValueError("Hypothesis unavailable after rejection or invalid design; "
                                         "explore another mechanism, not the same experiment")
                    designer = ResearchAgent(model)
                    designer.role = "research-designer"
                    designer.instruction = (
                        "Design a machine-executable Experiment Specification for THIS hypothesis. "
                        "You choose group/control, outcome, study unit, sample filter, strata, "
                        "optional negative-control calculation, a precise falsifier and meaningful "
                        "effect margin. Do NOT predefine Bias categories. Source_query optionally "
                        "calculates derived columns using a single SELECT or WITH expression. "
                        "Use the real schema and REGISTERED derived columns only. "
                        "With source_query='' Workbench uses the development view with "
                        "registered features, so use these features directly in group_expression, "
                        "strata_expression or population_filter. Do not invent aliases. "
                        "outcome_expression is a SINGLE per-event scalar, not AVG(), "
                        "not SELECT expression lists, and never AS alias: Workbench "
                        "aggregates over users. A custom source_query preserves uid and event rows. "
                        "Do not treat same-time log records as sequential state updates. "
                        "Prefer paired_user when both groups exist per user, otherwise "
                        "cluster_weighted. Optional SQL fields may be empty. "
                        "If you have no separate scientifically valid negative-control "
                        "assignment, set BOTH placebo fields to empty strings. "
                        "Never copy the treatment group into the placebo.")
                    designer.instruction += (
                        BIAS_RESEARCH_CONTRACT
                        + " This test must answer a concrete bias research "
                        "question, not only a valid statistical comparison. "
                        "Specify the decision stage, distorted quantity, "
                        "reference eligibility and assumptions, strongest "
                        "user-preference rival, comparable H1/H2 predictions, "
                        "estimand, user unit, negative control if justified "
                        "and what result would falsify the mechanism. "
                        "When the reference is unidentified, design a "
                        "reference-support probe, not a fake causal test."
                    )
                    event_days = tools.connection.execute(
                        "SELECT MIN(event_day), MAX(event_day) FROM events").fetchone()
                    first_workbench = not any(
                        "workbench" in test for test in selected["tests"])
                    feature_pairs = [(table, name) for table, items in tools.features.items()
                                     for name in items]
                    choice_contract = opportunity_pool.records.get(
                        selected["id"], {}).get("selection_contract")
                    from rsi_researcher.research.discovery.experiment_contract import (
                        experiment_intent_schema, compile_experiment_intent)
                    typed_design = bool(model_config.get("llm") and first_workbench
                                        and (feature_pairs or choice_contract)
                                        and discovery_options.get("compact_experiment", True))
                    design_schema = (experiment_intent_schema(tools, choice_contract)
                                     if typed_design else EXPERIMENT_SCHEMA)
                    if typed_design:
                        # The typed path is the default for real Qwen runs.
                        # Never replace its bias-science task with SQL-only guidance.
                        designer.instruction = (
                            "You design discriminating experiments for previously "
                            "unspecified recommender bias mechanisms. "
                            + BIAS_RESEARCH_CONTRACT
                            + " For the selected mechanism, first state in the "
                            "prediction and falsifier what RECOMMENDER decision "
                            "and reference discrepancy H1/H2 differ on. "
                            "Choose a measured registered group_feature, actual "
                            "control/treatment values, a numeric per-event outcome, "
                            "eligible population and user-level analysis. Python "
                            "compiles the SQL from this concise specification. "
                            "If the reference is unidentified, this is a "
                            "reference-support test only, not causal bias proof. "
                            "Do not hallucinate arm labels; random-vs-standard CTR "
                            "by itself is not item allocation bias.")
                    if first_workbench and feature_pairs and not typed_design:
                        design_schema = {**EXPERIMENT_SCHEMA, "properties": {
                            **EXPERIMENT_SCHEMA["properties"],
                            "source_query": {"type": "string", "enum": [""]},
                            "strata_expression": {"type": "string", "enum": sorted(
                                set(option for table, _ in feature_pairs
                                    for option in workbench.observed_strata_columns(table)))},
                            "source_table": {"type": "string",
                                             "enum": sorted(set(table for table, _ in feature_pairs))},
                            "group_expression": {"type": "string",
                                                 "enum": [name for _, name in feature_pairs]}}}
                    generated_spec = designer.act({
                        "available_event_day_range": list(event_days),
                        "population_filter_syntax": "Boolean SQL condition only, no leading WHERE",
                        "previous_execution_error": history[-1].get("tool_error") if history else None,
                        "concept": _concept_summary(selected),
                        "claim": selected["claim"],
                        "research_opportunity": opportunity_pool.records.get(selected["id"]),
                        "measured_selection_contract": opportunity_pool.records.get(
                            selected["id"], {}).get("selection_contract"),
                        "design_priority": "Choose the supported counterfactual observation "
                            "that makes the mechanism and rival predict DIFFERENT outcomes; "
                            "not the largest observed association.",
                        "schema": tools.schema(), "created_features": tools.features,
                        "registered_variable_cards": tools.feature_cards(),
                        "binding_instruction": "Bind experiments to registered feature NAME, "
                           "not a newly copied CASE expression.",
                        "registered_feature_observations": [
                            compact_step(item)["derived_feature"]
                            for item in history if "derived_feature" in item],
                        "example_data_values": data_examples,
                        "observation_cards": observation_cards,
                        "critical_design_rule": "A binary group must have observed users in BOTH actual arms; opaque tags are numeric IDs, and click/hate are current outcomes, not tag names.",
                        "prior_experiments": [
                            {"question": t["test_question"],
                             "numerical_evidence": compact_query(t["result"])}
                            for t in selected["tests"][-3:]],
                        "scientist_request": decision["reason"],
                    }, "experiment-design-" + str(step), design_schema)
                    experiment_spec = (compile_experiment_intent(
                        generated_spec, tools, choice_contract)
                        if typed_design else generated_spec)
                    # Give the Designer an explicit opportunity to cite its own
                    # measured variable instead of silently repeating the same
                    # CASE expression in each experiment plan.
                    registered = [name for values in tools.features.values()
                                  for name in values]
                    specified = " ".join(str(experiment_spec[key]) for key in
                                         ("source_query", "population_filter",
                                          "group_expression", "outcome_expression",
                                          "strata_expression"))
                    if registered and not any(re.search(
                            r"\b" + re.escape(name) + r"\b", specified)
                            for name in registered):
                        referenced = designer.act({
                            "previous": experiment_spec,
                            "registered_features": tools.features,
                            "real_schema": tools.schema(),
                            "request": "Use a REGISTERED derived feature by its COLUMN NAME "
                                       "in the group_expression, population_filter, "
                                       "strata_expression or source_query. You already "
                                       "computed and verified it; do not inline its CASE "
                                       "again. Preserve your scientific comparison and "
                                       "one per-event numeric outcome.",
                        }, "experiment-reuse-feature-" + str(step), EXPERIMENT_SCHEMA)
                        if any(re.search(r"\b" + re.escape(name) + r"\b",
                                      " ".join(str(referenced[k]) for k in
                                               ("source_query", "population_filter",
                                                "group_expression", "strata_expression")))
                               for name in registered):
                            experiment_spec = referenced
                            entry["feature_reuse_design"] = True
                        else:
                            entry["feature_reuse_warning"] = (
                                "Model did not reference any registered feature")
                    contrast_keys = ("source_query", "population_filter",
                                     "group_expression", "control_value",
                                     "treatment_value", "outcome_expression")
                    if any(all(previous["workbench"]["specification"][k] == experiment_spec[k]
                               for k in contrast_keys)
                           for previous in selected["tests"] if "workbench" in previous):
                        raise ValueError("Repeated numerical estimand for this hypothesis. "
                                         "Choose a different condition, outcome or mechanism")
                    choice_contract = opportunity_pool.records.get(
                        selected["id"], {}).get("selection_contract")
                    if choice_contract:
                        match_fields = ("source_table", "group_expression",
                                        "outcome_expression", "control_value",
                                        "treatment_value", "analysis")
                        entry["selection_contract_audit"] = {
                            "matches_original_contrast": all(
                                experiment_spec[k] == choice_contract[k]
                                for k in match_fields),
                            "differences": {
                                k: [choice_contract[k], experiment_spec[k]]
                                for k in match_fields if
                                experiment_spec[k] != choice_contract[k]},
                        }
                    import duckdb
                    entry["experiment_specification"] = experiment_spec
                    required_feature = (
                        experiment_spec["group_expression"]
                        if experiment_spec["group_expression"] in tools.feature_records
                        else next(reversed(tools.feature_records), None))
                    def validate_selected_contrast(specification):
                        if choice_contract:
                            keys = ("source_table", "group_expression", "outcome_expression",
                                    "control_value", "treatment_value", "analysis")
                            changed = [k for k in keys if
                                       specification[k] != choice_contract[k]]
                            if changed:
                                raise ValueError("Experiment silently changed the selected "
                                                 "rival-discriminating comparison: " +
                                                 ", ".join(changed) +
                                                 ". Use the prevalidated selection contract.")
                    try:
                        validate_selected_contrast(experiment_spec)
                        measured_experiment = workbench.run(
                            experiment_spec, required_feature=required_feature)
                    except (ValueError, duckdb.Error) as error:
                        corrected = designer.act({
                            "original": experiment_spec,
                            "execution_error": str(error),
                            "created_features": tools.features,
                            "registered_variable_cards": tools.feature_cards(),
                            "must_bind_feature": required_feature or "",
                            "selected_contrast_must_match": choice_contract,
                            "schema": tools.schema(),
                            "observed_rows": data_examples,
                            "instruction": "Repair the actual experiment to use TWO SUPPORTED "
                                           "groups and real row-level fields. Change the group "
                                           "or population, not just the hypothesis wording.",
                        }, "experiment-repair-" + str(step), design_schema)
                        if typed_design:
                            corrected = compile_experiment_intent(
                                corrected, tools, choice_contract)
                        if corrected == experiment_spec:
                            raise ValueError("Designer repeated an invalid experiment")
                        entry["experiment_repair"] = {"old": experiment_spec,
                                                       "error": str(error)}
                        experiment_spec = corrected
                        validate_selected_contrast(experiment_spec)
                        measured_experiment = workbench.run(
                            experiment_spec, required_feature=required_feature)
                    # Auditing must use the *executed* repaired spec, not a
                    # superseded first draft that may have been rejected.
                    if choice_contract:
                        material = ("source_table", "group_expression",
                                    "outcome_expression", "control_value",
                                    "treatment_value", "analysis")
                        entry["selection_contract_audit"] = {
                            "matches_original_contrast": all(
                                experiment_spec[key] == choice_contract[key]
                                for key in material),
                            "differences": {
                                key: [choice_contract[key], experiment_spec[key]]
                                for key in material if
                                experiment_spec[key] != choice_contract[key]},
                        }
                    entry["experiment"] = measured_experiment
                    reference_choice = opportunity_pool.records.get(
                        selected["id"], {}).get("bias_relevance", {})
                    selected_choice = next((
                        old["opportunity_selection"] for old in reversed(history)
                        if old.get("opportunity_selection", {}).get(
                            "selected_concept_id") == selected["id"]), None)
                    if (selected_choice and reference_choice.get("reference_kind")
                            == "randomized_exposure"):
                        try:
                            from rsi_researcher.research.discovery.workbench import randomized_reference_probe
                            reference = randomized_reference_probe(tools, selected_choice)
                            entry["randomized_reference_diagnostic"] = reference
                            opportunity_pool.record_reference_diagnostic(selected["id"], reference)
                        except Exception as exc:
                            entry["randomized_reference_probe_error"] = str(exc)[:350]
                    entry["feature_binding"] = {
                        "registered_feature": required_feature,
                        "original_expression": (
                            tools.feature_records[required_feature]["expression"]
                            if required_feature else None),
                        "executed_sql": measured_experiment["executed_sql"]}
                    entry["feature_lineage"] = {
                        "used": [name for _, name in feature_pairs
                                 if re.search(r"\b" + re.escape(name) + r"\b",
                                              " ".join(str(experiment_spec[k]) for k in
                                              ("source_query", "population_filter",
                                               "group_expression", "outcome_expression",
                                               "strata_expression")))],
                        "source_tables": sorted(set(table for table, _ in feature_pairs)),
                    }
                    numeric = {"sql": measured_experiment["sql"],
                               "columns": measured_experiment["columns"],
                               "rows": measured_experiment["rows"]}
                    if opportunity_pool.focus != selected["id"]:
                        opportunity_pool.choose(
                            selected["id"], decision["reason"] or
                            "Agent selected this opportunity for a statistical experiment")
                    selected["tests"].append({
                        "version": selected["version"],
                        "test_question": experiment_spec["question"],
                        "predicted_falsifier": experiment_spec["falsifier"],
                        "result": numeric, "workbench": measured_experiment})
                    critique = critic.act({
                        "concept": _concept_summary(selected),
                        "hypothesis": selected["claim"],
                        "research_opportunity": opportunity_pool.records.get(selected["id"]),
                        "experiment_specification": experiment_spec,
                        "observed_statistics": model_statistics(measured_experiment["result"]),
                        "randomized_reference_diagnostic": entry.get(
                            "randomized_reference_diagnostic"),
                        "negative_control": (model_statistics(
                            measured_experiment["negative_control"]["result"])
                            if measured_experiment["negative_control"] else None),
                        "request": (
                            "Evaluate distortion measurement, reference "
                            "identification, H1/H2 numerical disagreement, "
                            "logging artifacts and consequence separately. "
                            "If any is unsupported, name exactly which "
                            "scientific requirement failed and choose a "
                            "reference-support or rival-discriminating test."),
                    }, "experiment-critique-" + str(step), CRITIC_SCHEMA)
                    entry["critique"] = critique
                    selected["critique"] = critique
                    scientific_gaps = scientific_evidence_gaps(
                        selected, opportunity_pool.records.get(selected["id"], {}))
                    entry["scientific_evidence_gaps"] = scientific_gaps
                    opportunity_pool.records[selected["id"]][
                        "scientific_evidence_gaps"] = scientific_gaps
                    if critique["decision"] == "test":
                        followup_generated = designer.act({
                            "concept": _concept_summary(selected),
                            "previous_experiment": experiment_spec,
                            "results": model_statistics(measured_experiment["result"]),
                            "placebo": (model_statistics(measured_experiment["negative_control"]["result"])
                                        if measured_experiment["negative_control"] else None),
                            "critic_feedback": critique,
                            "schema": tools.schema(),
                            "available_event_day_range": list(event_days),
                            "instruction": "Propose a different experiment, grouping, control, time window or placebo that distinguishes the rival. Do not repeat the earlier specification.",
                        }, "experiment-countertest-" + str(step),
                            experiment_intent_schema(tools, choice_contract)
                            if typed_design else EXPERIMENT_SCHEMA)
                        followup_spec = (compile_experiment_intent(
                            followup_generated, tools, choice_contract)
                            if typed_design else followup_generated)
                        # Reformatting the prompt or adding strata alone
                        # does not produce a different primary numeric contrast.
                        material = ("source_query", "population_filter",
                                    "group_expression", "control_value",
                                    "treatment_value", "outcome_expression")
                        if all(followup_spec[k] == experiment_spec[k] for k in material):
                            # One controlled model-chosen mutation is easier for an
                            # 8B model than regenerating 17 interdependent fields.
                            supported_filters = feasible_countertest_filters(
                                tools, experiment_spec)
                            delta_schema = COUNTER_DELTA_SCHEMA
                            if supported_filters:
                                delta_schema = {**COUNTER_DELTA_SCHEMA, "properties": {
                                    **COUNTER_DELTA_SCHEMA["properties"],
                                    "field": {"type": "string", "enum": ["population_filter"]},
                                    "replacement": {"type": "string", "enum": [
                                        item["population_filter"] for item in supported_filters]}}}
                            delta = designer.act({
                                "feasible_development_subgroups": supported_filters,
                                "original": experiment_spec,
                                "critic_reason": critique["reason"],
                                "critic_sql": critique["next_sql"],
                                "observed_statistics": model_statistics(
                                    measured_experiment["result"]),
                                "schema": tools.schema(),
                                "registered_features": tools.features,
                                "task": "Change ONE experimental contrast dimension. "
                                        "Choose a DIFFERENT supported user subgroup, "
                                        "grouping, outcome or effect modifier. "
                                        "Do not repeat the original string. "
                                        "Keep unit_expression, data table, group labels "
                                        "and inferential analysis unchanged.",
                            }, "experiment-countertest-delta-" + str(step),
                                delta_schema)
                            followup_spec = dict(experiment_spec)
                            followup_spec[delta["field"]] = delta["replacement"]
                            entry["countertest_delta"] = delta
                        if followup_spec == experiment_spec:
                            # The Critic's concrete falsification SQL remains independently
                            # selectable even if the Designer fails to modify a whole plan.
                            # It is exploratory and never a substitute for a paired CI.
                            entry["critic_experiment_error"] = "Designer repeated the experiment"
                            if critique["next_sql"].strip():
                                try:
                                    entry["critic_sql_countertest"] = tools.sql(critique["next_sql"])
                                except Exception as exc:
                                    entry["critic_sql_error"] = str(exc)[:280]
                        else:
                            try:
                                try:
                                    followup = workbench.run(followup_spec)
                                except (ValueError, duckdb.Error) as error:
                                    repaired = designer.act({
                                        "failed_countertest": followup_spec,
                                        "execution_error": str(error),
                                        "original_experiment": experiment_spec,
                                        "registered_features": tools.features,
                                        "exact_schema": tools.schema(),
                                        "instruction": "Repair the COUNTERTEST, not the original. "
                                                       "Replace nonexistent columns with real "
                                                       "registered names, ensure different group/"
                                                       "population/stratum and both comparison "
                                                       "groups have user-level support. "
                                                       "Return ONE per-event numeric outcome.",
                                    }, "countertest-execution-repair-" + str(step),
                                        EXPERIMENT_SCHEMA)
                                    if repaired == followup_spec or repaired == experiment_spec:
                                        raise ValueError("No distinct corrected countertest")
                                    followup_spec = repaired
                                    followup = workbench.run(followup_spec)
                                    entry["critic_experiment_repaired"] = str(error)[:260]
                                entry["critic_experiment"] = followup
                                selected["tests"].append({
                                    "version": selected["version"],
                                    "test_question": followup_spec["question"],
                                    "predicted_falsifier": followup_spec["falsifier"],
                                    "requested_by": "critic",
                                    "result": {"sql": followup["sql"],
                                               "columns": followup["columns"],
                                               "rows": followup["rows"]},
                                    "workbench": followup})
                            except Exception as exc:
                                entry["critic_experiment_error"] = str(exc)[:300]
                    if critique["decision"] == "reject":
                        selected["status"] = "rejected_development"
                    elif critique["decision"] == "revise":
                        selected["status"] = "needs_revision"
                    # A countertest that does not resolve the rival forces a real
                    # next-research choice rather than reusing the same hypothesis.
                    if (critique["decision"] == "test"
                            and ("critic_experiment" in entry
                                 or "critic_sql_countertest" in entry)
                            and selected["status"] != "rejected_development"):
                        # A Critic-authored SQL probe is legitimate numerical
                        # falsification evidence but has no user-level uncertainty;
                        # it must NEVER be presented as a Workbench paired test.
                        counter_result = (
                            model_statistics(entry["critic_experiment"]["result"])
                            if "critic_experiment" in entry else
                            {"descriptive_sql_only": True,
                             "observations": compact_query(entry["critic_sql_countertest"])})
                        reflector = ResearchAgent(model)
                        reflector.role = "discovery-reflection"
                        reflector.instruction = (
                            "You have an initial statistical experiment AND an actually executed "
                            "Critic countertest. Distinguish descriptive SQL from paired inferential "
                            "statistics; their NUMERIC results are authoritative within scope. "
                            "Decide whether to revise the mechanism/measurement, abandon "
                            "or request literature. Merely renaming the same hypothesis "
                            "is NOT revision. State a falsifiable alternative.")
                        reflection_input = {
                            "concept": selected["claim"],
                            "initial_result": model_statistics(measured_experiment["result"]),
                            "countertest_result": counter_result,
                            "critique": critique,
                        }
                        reflector.instruction += BIAS_RESEARCH_CONTRACT
                        reflection_input["bias_relevance"] = opportunity_pool.records.get(
                            selected["id"], {}).get("bias_relevance", {})
                        reflection_input["scientific_evidence_gaps"] = (
                            scientific_evidence_gaps(
                                selected, opportunity_pool.records.get(
                                    selected["id"], {})))
                        reflection_input["next_research_requirement"] = (
                            "Identify whether a reference, discrimination or "
                            "experiment-support failure caused this result; "
                            "change the next measurable test accordingly."
                        )
                        reflection = reflector.act(
                            reflection_input, "experiment-reflection-" + str(step),
                            REFLECTION_SCHEMA)
                        if (reflection["decision"] == "revise"
                                and all(reflection[k].strip() == selected["claim"][k].strip()
                                        for k in ("falsifier", "measurement"))):
                            entry["rejected_cosmetic_reflection"] = reflection
                            # Scientific invalidity should cause a clear stop,
                            # not infinite rewording of the original hypothesis.
                            reflection = reflector.act({
                                **reflection_input,
                                "invalid_previous_revision": reflection,
                                "instruction": "Your proposed revision was only a rephrasing. "
                                               "Either ABANDON this unsupported mechanism "
                                               "or specify a DIFFERENT falsifiable measurement "
                                               "and an executable SQL query that distinguishes "
                                               "it from the rival. Do not invent a result.",
                            }, "experiment-reflection-recheck-" + str(step),
                                REFLECTION_SCHEMA)
                        entry["reflection"] = reflection
                        revised = any(
                            reflection[key].strip() != selected["claim"][key].strip()
                            for key in ("falsifier", "measurement"))
                        if reflection["decision"] == "revise" and revised:
                            try:
                                if not reflection["next_sql"].strip():
                                    raise ValueError("Revision needs an executable new falsification query")
                                entry["revision_probe"] = tools.sql(reflection["next_sql"])
                            except Exception as exc:
                                entry["reflection_veto"] = ("Proposed revised measurement did not execute: "
                                                            + str(exc)[:250])
                                selected["status"] = "needs_revision"
                            else:
                                selected["revisions"].append({
                                    "previous": selected["claim"],
                                    "reason": reflection["reason"],
                                    "triggering_tests": len(selected["tests"])})
                                selected["claim"] = {
                                    **selected["claim"],
                                    **{k: reflection[k] for k in (
                                        "concept", "mechanism", "rival", "falsifier", "measurement")},
                                    "query": reflection["next_sql"], "reason": reflection["reason"]}
                                selected["version"] += 1
                                selected["critique"] = None
                                entry["revised_concept"] = _concept_summary(selected)
                        elif reflection["decision"] == "revise":
                            entry["reflection_veto"] = "No substantive change of mechanism, rival, falsifier or metric"
                            selected["status"] = "needs_revision"
                        elif reflection["decision"] == "abandon":
                            selected["status"] = "rejected_development"
                        elif reflection["decision"] == "literature":
                            selected["literature"] = literature_review(
                                model, selected["claim"], "experiment-literature-" + str(step))
                            entry["literature_summary"] = selected["literature"]["assessment"]
                elif act == "revise":
                    if selected is None:
                        raise ValueError("Select a concept_id for revision")
                    if not all(decision[k].strip() for k in
                               ("concept", "mechanism", "rival", "falsifier", "measurement")):
                        raise ValueError("Revisions need a complete falsifiable concept")
                    if not selected["tests"]:
                        raise ValueError("A revision needs executed contradictory evidence")
                    if all(decision[k].strip() == selected["claim"][k].strip()
                           for k in ("falsifier", "measurement")):
                        raise ValueError("Revision must change the falsifier or measurement")
                    if not decision["query"].strip():
                        raise ValueError("Revision must include a new executable measurement")
                    entry["revision_probe"] = tools.sql(decision["query"])
                    selected["revisions"].append({
                        "previous": selected["claim"], "reason": decision["reason"],
                        "triggering_tests": len(selected["tests"])})
                    selected.update({"claim": decision, "version": selected["version"] + 1,
                                     "status": "investigating", "critique": None})
                    entry["revised_concept"] = _concept_summary(selected)
                elif act == "literature":
                    if selected is None:
                        raise ValueError("Form a concept before checking the literature")
                    selected["literature"] = literature_review(
                        model, selected["claim"], "literature-" + str(step))
                    entry["literature_summary"] = selected["literature"]["assessment"]
                elif act == "freeze":
                    if selected is None:
                        raise ValueError("Select an existing concept to freeze")
                    numeric_ref = opportunity_pool.records.get(
                        selected["id"], {}).get("bias_relevance", {}).get(
                            "numeric_discrepancy", {})
                    ci = numeric_ref.get("ci95") or []
                    audit = {
                        "reference_gap_ci_excludes_zero": bool(
                            len(ci) == 2 and (ci[0] > 0 or ci[1] < 0)),
                        "two_exploratory_queries": len(tools.history) >= min_queries,
                        "has_development_test": bool(selected["tests"]),
                        "critic_ready": (selected["critique"] or {}).get("decision") == "ready",
                        "executed_rival_countertest": any(
                            t.get("requested_by") == "critic" and "workbench" in t
                            for t in selected["tests"]),
                        "observable_valid": bool((selected["critique"] or {}).get("observable_valid")),
                        "rival_tested": bool((selected["critique"] or {}).get("rival_discriminated")),
                        "logging_checked": bool((selected["critique"] or {}).get("logging_artifact_excluded")),
                        "literature_checked": selected["literature"] is not None,
                        "reference_target_declared": bool(
                            opportunity_pool.records.get(selected["id"], {}).get(
                                "bias_relevance", {}).get("target_reference")),
                        "numeric_reference_gap_computed": bool(
                            opportunity_pool.records.get(selected["id"], {}).get(
                                "bias_relevance", {}).get("numeric_discrepancy", {}).get("measured")),
                        "credible_normative_target_validated": bool(
                            opportunity_pool.records.get(selected["id"], {}).get(
                                "bias_relevance", {}).get("numeric_discrepancy", {}).get(
                                    "normative_target_validated")),
                    }
                    entry["freeze_gate"] = audit
                    if all(audit.values()):
                        selected["status"] = "frozen_unconfirmed"
                        last = selected["id"]
                        history.append(entry)
                        break
                    entry["gate"] = "not ready; choose another research action"
                elif act == "stop":
                    stop_reason = "scientist_stop: " + decision["reason"]
                    entry["termination_reason"] = stop_reason
                    history.append(entry)
                    break
            except Exception as exc:
                entry["tool_error"] = str(exc)[:400]
                if act == "experiment" and selected is not None:
                    failures = experiment_failures.get(selected["id"], 0) + 1
                    experiment_failures[selected["id"]] = failures
                    if failures >= 2:
                        selected["status"] = "design_blocked"
                        entry["design_status"] = "failed_two_real_experiment_designs"
                        entry["next_research_instruction"] = (
                            "Choose another mechanism or create a new observable "
                            "item-side feature; do not repeat unsupported arms.")
            if selected is not None and selected["id"] in opportunity_pool.records:
                if selected.get("critique") is not None:
                    gaps = scientific_evidence_gaps(
                        selected, opportunity_pool.records[selected["id"]])
                    entry["scientific_evidence_gaps"] = gaps
                    opportunity_pool.records[selected["id"]][
                        "scientific_evidence_gaps"] = gaps
                if "experiment" in entry and "critique" in entry:
                    opportunity_pool.experiment(
                        selected["id"], entry["experiment"],
                        entry.get("critic_experiment"), entry["critique"])
                if ("reflection" in entry or "revised_concept" in entry
                        or selected["status"] in ("rejected_development", "design_blocked")):
                    opportunity_pool.update_claim(
                        selected["id"], selected["claim"],
                        ("abandoned" if selected["status"] == "rejected_development"
                         else selected["status"]))
                opportunity_pool.record_research_decision(
                    selected["id"], selected, entry)
            history.append(entry)
            # New observations, measured contrasts or a corrected mechanism
            # are information; merely naming another hypothesis or retrying
            # invalid SQL is not. The stall guard is NOT a total research budget.
            evidence_after = research_evidence_signature(tools, concepts, opportunity_pool)
            no_information_actions = (
                0 if evidence_after != evidence_before
                else no_information_actions + 1)
            _save(dest / "research_opportunities.json", opportunity_pool.snapshot())
            _save(dest / "journal.json", history)
            _save(dest / "research_state.json", {
                "concepts": concepts, "rounds_completed": len(history),
                "registered_features": tools.features,
                "no_information_actions": no_information_actions,
                "holdout_consulted": False})
            if max_rounds is None and no_information_actions >= 6:
                stop_reason = (
                    "scientific_stall: six consecutive actions produced no "
                    "new observation, measured contrast or substantive revision")
                history.append({
                    "round": step, "action": {"action": "research_stalled",
                        "reason": stop_reason, "concept_id": "",
                        "query": "", "concept": ""},
                    "termination_reason": stop_reason})
                break
        status = ("frozen_unconfirmed" if last else
                  "development_inconclusive" if max_rounds is not None else
                  "scientific_stalled" if stop_reason.startswith("scientific_stall") else
                  "scientific_stop" if stop_reason else "development_inconclusive")
        final = {"status": status, "frozen_id": last or None,
                 "concepts": concepts, "registered_features": tools.features,
                 "development_queries": tools.history,
                 "observation_cards": observation_cards,
                 "research_opportunities": opportunity_pool.snapshot(),
                 "research_rounds": len(history), "holdout_consulted": False,
                 "termination_reason": stop_reason or (
                     "fixed_round_limit" if max_rounds is not None and not last else ""),
                 "independent_confirmation": False,
                 "causal_bias_established": False, "debias_verified": False,
                 "research_funnel": research_funnel(concepts, opportunity_pool)}
        _save(dest / "journal.json", history)
        _save(dest / "research_state.json", {
            "concepts": concepts, "rounds_completed": len(history),
            "termination_reason": final["termination_reason"],
            "no_information_actions": no_information_actions,
            "holdout_consulted": False})
        _save(dest / "frozen_candidate.json", final)
        return final
    except (BudgetExceeded, ModelFailure, KeyboardInterrupt) as error:
        # Preserve the observed science state on an operator interrupt,
        # bounded-comparison quota or Qwen transport failure.
        reason = ("interrupted" if isinstance(error, KeyboardInterrupt) else
                  "budget_exhausted" if isinstance(error, BudgetExceeded)
                  else "model_failed")
        history.append({"round": len(history),
                        "action": {"action": "runtime_stopped",
                                   "reason": reason, "concept_id": "",
                                   "query": "", "concept": ""},
                        "runtime_error": str(error)[:280]})
        final = {
            "status": reason, "frozen_id": None, "concepts": concepts,
            "registered_features": tools.features,
            "development_queries": tools.history,
            "research_opportunities": opportunity_pool.snapshot(),
            "research_rounds": len(history), "holdout_consulted": False,
            "independent_confirmation": False,
            "causal_bias_established": False, "debias_verified": False,
            "research_funnel": research_funnel(concepts, opportunity_pool)}
        _save(dest / "journal.json", history)
        _save(dest / "research_opportunities.json", opportunity_pool.snapshot())
        _save(dest / "research_state.json", {"concepts": concepts,
            "rounds_completed": len(history), "registered_features": tools.features,
            "holdout_consulted": False})
        _save(dest / "frozen_candidate.json", final)
        return final
    finally:
        tools.close()
