"""Model-directed, evidence-first research loop on physically isolated development data.

Exploration can define variables, inspect trajectories, form and revise concepts,
run discrimination tests, consult literature, and freeze an independently testable
proposal. Only the evaluator (outside this module) may touch holdout.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from rsi_researcher.research.discovery.opportunities import ResearchOpportunityPool
from rsi_researcher.research.scientific_agents import ResearchAgent
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.models import create_model

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
            "falsifier": TEXT, "measurement": TEXT, "suggested_sql": TEXT},
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
    },
    "required": ["selected_concept_id", "reason", "discriminating_prediction",
                 "prediction_if_mechanism", "prediction_if_rival",
                 "why_not_other", "next_action", "next_sql", "experimental_cost"],
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
        con.execute("""CREATE TABLE events AS SELECT
            uid,vid,event_day,moment_ms,is_random,click,hate,
            long_view,tags,previous_video,previous_click,
            previous_hate,previous_moment,previous_tags,next_tags
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
        if re.search(r"\b(click|hate|long_view|next_click)\b", expression, re.I):
            raise ValueError("Do not build a predictor from the CURRENT outcome label; use "
                             "previous_click/previous_hate, timestamps or other observed "
                             "pre-event fields, and measure current outcomes separately")
        # Execute the definition on the chosen source before registering a view.
        evidence = self.sql(
            "SELECT (" + expression + ") AS " + name + ",count(*) n FROM " +
            source_table + " GROUP BY 1 ORDER BY n DESC LIMIT 8")
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
        rows = self.connection.execute(
            "SELECT " + ",".join(fields) + " FROM events LIMIT 8").fetchall()
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


def _save(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run_open_discovery(source_db: str, output: str, model_config: dict,
                       max_rounds: int = 12, min_queries: int = 2,
                       development_path: str | None = None) -> dict:
    """Run a reproducible multi-role active research episode; no holdout reads."""
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
    from rsi_researcher.research.discovery.workbench import (
        ExperimentalWorkbench, EXPERIMENT_SCHEMA)
    workbench = ExperimentalWorkbench(tools)
    model_dir = dest / "model_calls"
    model_dir.mkdir()
    limits = RunLimits(model_dir, max_requests=3 * max_rounds + 4,
                       max_tokens=250000, max_seconds=2400)
    model = create_model(model_dir, limits.max_requests, limits, model_config)
    scientist = ResearchAgent(model)
    scientist.role = "open-discovery-scientist"
    scientist.instruction = (
        "You are an autonomous recommender-systems research scientist, not a "
        "candidate classifier. Investigate user and recommendation SEQUENCES. "
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
    history, concepts = [], []
    opportunity_pool = ResearchOpportunityPool()
    last = ""
    try:
        for step in range(max_rounds):
            if step >= 3 and not concepts and tools.history:
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
                    "Invent conceptual relationships, not old named Bias cards. "
                    "For each concept specify an observable prediction if the mechanism "
                    "is true, a DIFFERENT observable prediction if its rival is true, "
                    "and what follow-up observation or experiment best separates them. "
                    "Use the actual measured rows; do not make up evidence, effect "
                    "sizes or sample counts. We want research opportunity selection, "
                    "not generic significant findings. "
                    "Do not claim causality or novelty. Use exact schema columns.")
                formed = former.act({
                    "observations": [compact_query(x) for x in tools.history[-4:]],
                    "tables": tools.schema(),
                    "existing_derived_features": tools.features,
                }, "concept-formation-" + str(step), CONCEPT_PAIR_SCHEMA)
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
                # consumes the same Qwen/tool budget. Test fixtures without an
                # actual configured LLM still exercise the normal action route.
                if model_config.get("llm"):
                    chooser = ResearchAgent(model)
                    chooser.role = "research-opportunity-selector"
                    chooser.instruction = (
                        "Choose which AGENT-PROPOSED phenomenon to investigate next. "
                        "Compare the mechanism-vs-rival predictions; select the "
                        "observation or statistical experiment where they disagree "
                        "most, taking real measured support and cost into account. "
                        "State distinct observable predictions for BOTH the proposed "
                        "mechanism and its rival; repeating the same prediction is "
                        "not an informative experiment. "
                        "Do NOT rank by significance alone, invent an additional "
                        "candidate, or use any hidden confirmation data. An observe "
                        "decision needs an executable development SQL query.")
                    choice = chooser.act({
                        "opportunities": opportunity_pool.for_agent(),
                        "recent_observations": [compact_query(
                            q) for q in tools.history[-4:]],
                        "available_schema": tools.schema(),
                        "budget_left": max_rounds - step,
                    }, "opportunity-choice-" + str(step),
                        OPPORTUNITY_CHOICE_SCHEMA)
                    opportunity_pool.choose(
                        choice["selected_concept_id"], choice["reason"],
                        choice["next_sql"] if choice["next_action"] == "observe" else "")
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
                        "opportunity_selection": choice}
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
                "derived_features": tools.features,
                "registered_variable_cards": tools.feature_cards(),
                "concepts": [_concept_summary(c) for c in concepts[-4:]],
                "research_opportunity_pool": opportunity_pool.for_agent(),
                "opportunity_selection_principle": (
                    "You own the next research choice: select_opportunity with a "
                    "concept_id and optionally a fresh SQL observation, or choose "
                    "SQL/define_feature/experiment directly. Prefer the question "
                    "whose next MEASURED outcome would distinguish competing "
                    "mechanisms. Explain the discriminating prediction, expected "
                    "information, observable support and cost. Do NOT prioritize "
                    "a hypothesis merely because its p-value or effect is large."),
                "recent_evidence": [compact_step(s) for s in history[-4:]],
                "steps_left": max_rounds - step,
                "successful_queries": len(tools.history),
                "minimum_queries_before_freeze": min_queries,
                "available_actions": ACTION_SCHEMA["properties"]["action"]["enum"],
                "research_mission": "Invent a testable mechanism not supplied as a candidate card.",
                "sql_hint": "For new concepts measure conditional interactions, with counts and user units.",
                "next_action_guidance": (
                    "Prioritize an actual test using EXACT H001/H002 IDs now; "
                    "then revise the mechanism if contradicted; request literature "
                    "before attempting freeze." if concepts else
                    "Inspect actual data, register a reusable row-level variable with "
                    "define_feature, then verify it with SQL before experimental design."
                    if not any(tools.features.values()) else
                    "Test the registered variable with a conditional comparison and "
                    "form a falsifiable hypothesis with a competing explanation."),
            }
            decision = scientist.act(instruction, "open-" + str(step), ACTION_SCHEMA)
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
                    event_days = tools.connection.execute(
                        "SELECT MIN(event_day), MAX(event_day) FROM events").fetchone()
                    first_workbench = not any(
                        "workbench" in test for test in selected["tests"])
                    feature_pairs = [(table, name) for table, items in tools.features.items()
                                     for name in items]
                    design_schema = EXPERIMENT_SCHEMA
                    if first_workbench and feature_pairs:
                        design_schema = {**EXPERIMENT_SCHEMA, "properties": {
                            **EXPERIMENT_SCHEMA["properties"],
                            "source_query": {"type": "string", "enum": [""]},
                            "source_table": {"type": "string",
                                             "enum": sorted(set(table for table, _ in feature_pairs))},
                            "group_expression": {"type": "string",
                                                 "enum": [name for _, name in feature_pairs]}}}
                    experiment_spec = designer.act({
                        "available_event_day_range": list(event_days),
                        "population_filter_syntax": "Boolean SQL condition only, no leading WHERE",
                        "previous_execution_error": history[-1].get("tool_error") if history else None,
                        "concept": _concept_summary(selected),
                        "claim": selected["claim"],
                        "research_opportunity": opportunity_pool.records.get(selected["id"]),
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
                        "critical_design_rule": "A binary group must have observed users in BOTH actual arms; opaque tags are numeric IDs, and click/hate are current outcomes, not tag names.",
                        "prior_experiments": [
                            {"question": t["test_question"],
                             "numerical_evidence": compact_query(t["result"])}
                            for t in selected["tests"][-3:]],
                        "scientist_request": decision["reason"],
                    }, "experiment-design-" + str(step), design_schema)
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
                    import duckdb
                    entry["experiment_specification"] = experiment_spec
                    required_feature = next(reversed(tools.feature_records), None)
                    try:
                        measured_experiment = workbench.run(
                            experiment_spec, required_feature=required_feature)
                    except (ValueError, duckdb.Error) as error:
                        corrected = designer.act({
                            "original": experiment_spec,
                            "execution_error": str(error),
                            "created_features": tools.features,
                            "registered_variable_cards": tools.feature_cards(),
                            "must_bind_feature": required_feature or "",
                            "schema": tools.schema(),
                            "observed_rows": data_examples,
                            "instruction": "Repair the actual experiment to use TWO SUPPORTED "
                                           "groups and real row-level fields. Change the group "
                                           "or population, not just the hypothesis wording.",
                        }, "experiment-repair-" + str(step), design_schema)
                        if corrected == experiment_spec:
                            raise ValueError("Designer repeated an invalid experiment")
                        entry["experiment_repair"] = {"old": experiment_spec,
                                                       "error": str(error)}
                        experiment_spec = corrected
                        measured_experiment = workbench.run(
                            experiment_spec, required_feature=required_feature)
                    entry["experiment"] = measured_experiment
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
                        "negative_control": (model_statistics(
                            measured_experiment["negative_control"]["result"])
                            if measured_experiment["negative_control"] else None),
                        "request": "Evaluate rival mechanism, logging artifacts, and whether a DIFFERENT experiment is needed.",
                    }, "experiment-critique-" + str(step), CRITIC_SCHEMA)
                    entry["critique"] = critique
                    selected["critique"] = critique
                    if critique["decision"] == "test":
                        followup_spec = designer.act({
                            "concept": _concept_summary(selected),
                            "previous_experiment": experiment_spec,
                            "results": model_statistics(measured_experiment["result"]),
                            "placebo": (model_statistics(measured_experiment["negative_control"]["result"])
                                        if measured_experiment["negative_control"] else None),
                            "critic_feedback": critique,
                            "schema": tools.schema(),
                            "available_event_day_range": list(event_days),
                            "instruction": "Propose a different experiment, grouping, control, time window or placebo that distinguishes the rival. Do not repeat the earlier specification.",
                        }, "experiment-countertest-" + str(step), EXPERIMENT_SCHEMA)
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
                    audit = {
                        "two_exploratory_queries": len(tools.history) >= min_queries,
                        "has_development_test": bool(selected["tests"]),
                        "critic_ready": (selected["critique"] or {}).get("decision") == "ready",
                        "observable_valid": bool((selected["critique"] or {}).get("observable_valid")),
                        "rival_tested": bool((selected["critique"] or {}).get("rival_discriminated")),
                        "logging_checked": bool((selected["critique"] or {}).get("logging_artifact_excluded")),
                        "literature_checked": selected["literature"] is not None,
                    }
                    entry["freeze_gate"] = audit
                    if all(audit.values()):
                        selected["status"] = "frozen_unconfirmed"
                        last = selected["id"]
                        history.append(entry)
                        break
                    entry["gate"] = "not ready; choose another research action"
                elif act == "stop":
                    history.append(entry)
                    break
            except Exception as exc:
                entry["tool_error"] = str(exc)[:400]
            if selected is not None and selected["id"] in opportunity_pool.records:
                if "experiment" in entry and "critique" in entry:
                    opportunity_pool.experiment(
                        selected["id"], entry["experiment"],
                        entry.get("critic_experiment"), entry["critique"])
                if ("reflection" in entry or "revised_concept" in entry
                        or selected["status"] == "rejected_development"):
                    opportunity_pool.update_claim(
                        selected["id"], selected["claim"],
                        ("abandoned" if selected["status"] == "rejected_development"
                         else selected["status"]))
            history.append(entry)
            _save(dest / "research_opportunities.json", opportunity_pool.snapshot())
            _save(dest / "journal.json", history)
            _save(dest / "research_state.json", {
                "concepts": concepts, "rounds_completed": len(history),
                "registered_features": tools.features,
                "holdout_consulted": False})
        status = "frozen_unconfirmed" if last else "development_inconclusive"
        final = {"status": status, "frozen_id": last or None,
                 "concepts": concepts, "registered_features": tools.features,
                 "development_queries": tools.history,
                 "research_opportunities": opportunity_pool.snapshot(),
                 "research_rounds": len(history), "holdout_consulted": False,
                 "independent_confirmation": False,
                 "causal_bias_established": False, "debias_verified": False}
        _save(dest / "journal.json", history)
        _save(dest / "research_state.json", {
            "concepts": concepts, "rounds_completed": len(history),
            "holdout_consulted": False})
        _save(dest / "frozen_candidate.json", final)
        return final
    finally:
        tools.close()
