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

from rsi_researcher.research.scientific_agents import ResearchAgent
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.models import create_model

TEXT = {"type": "string"}
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [
            "sql", "inspect_trajectory", "define_feature", "form_concept",
            "test", "experiment", "revise", "literature", "freeze", "stop"]},
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
        executed = clean
        # Transparently expose derived columns through the original table names.
        # Keep base tables immutable and avoid recursive shadow views.
        for base, fields in self.features.items():
            if fields:
                executed = re.sub(
                    r"\b(FROM|JOIN)\s+" + base + r"\b",
                    lambda m: m.group(1) + " research_" + base, executed,
                    flags=re.I)
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

    def define_feature(self, name: str, expression: str, source_table: str = "features"):
        if source_table not in ("features", "events"):
            raise ValueError("Derived feature source_table must be features or events")
        if not IDENTIFIER.fullmatch(name) or name in [c["column"] for c in self.schema()[source_table]]:
            raise ValueError("Feature name must be a fresh lowercase SQL identifier")
        if not expression.strip() or len(expression) > 1000 or ";" in expression or FORBIDDEN.search(expression):
            raise ValueError("Feature expression must be an SQL scalar expression")
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
        return {"name": name, "expression": expression, "evidence": evidence,
                "usable_table": source_table, "explicit_view": view,
                "source_table": source_table}

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

    def close(self):
        self.connection.close()


def compact_query(record: dict):
    return {"sql": record["sql"][:500],
            "executed_sql": record.get("executed_sql", record["sql"])[:500],
            "columns": record["columns"][:15],
            "rows": record["rows"][:8],
            "returned_rows": len(record["rows"]),
            "model_excerpt_only": len(record["rows"]) > 8}


def compact_step(entry: dict):
    action = entry["action"]
    result = {"round": entry["round"], "action": action["action"],
              "reason": action["reason"][:200], "concept_id": action["concept_id"],
              "concept": action["concept"][:170]}
    for k in ("result", "test", "critic_test"):
        if k in entry:
            result[k] = compact_query(entry[k])
    if "trajectory" in entry:
        result["trajectory"] = compact_query(entry["trajectory"]["trajectory"])
    if "experiment" in entry:
        record = entry["experiment"]
        result["experiment"] = {
            "question": record["specification"]["question"],
            "specification": record["specification"],
            "result": record["result"],
            "negative_control": (record["negative_control"] or {}).get("result"),
        }
    if "critic_experiment" in entry:
        result["critic_experiment"] = entry["critic_experiment"]["result"]
    if "derived_feature" in entry:
        result["derived_feature"] = {
            "name": entry["derived_feature"]["name"],
            "expression": entry["derived_feature"]["expression"]}
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
                    stage_ids.append(identifier)
                stage_action = {"action": "concept_formation", "reason": "hypothesis synthesis",
                                "concept_id": "", "concept": "", "query": ""}
                history.append({"round": step, "action": stage_action,
                                "concept_formation": stage_ids})
                _save(dest / "concept_formation.json", formed)
                _save(dest / "journal.json", history)
            instruction = {
                "tables": tools.schema(),
                "derived_features": tools.features,
                "concepts": [_concept_summary(c) for c in concepts[-4:]],
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
                    "Explore informative trajectories and define reusable features "
                    "before concept formation. Do not just select known Bias names."),
            }
            decision = scientist.act(instruction, "open-" + str(step), ACTION_SCHEMA)
            entry = {"round": step, "action": decision}
            act = decision["action"]
            selected = next((c for c in concepts if c["id"] == decision["concept_id"]), None)
            try:
                if act == "sql":
                    entry["result"] = tools.sql(decision["query"])
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
                        "Use actual available schema; do not confuse same-time batched exposure "
                        "with sequential recommender updates. Prefer user_paired when both groups "
                        "exist per user; otherwise cluster_weighted. Empty optional SQL fields "
                        "are empty strings. Placebo group expression may be empty if unjustified.")
                    event_days = tools.connection.execute(
                        "SELECT MIN(event_day), MAX(event_day) FROM events").fetchone()
                    experiment_spec = designer.act({
                        "available_event_day_range": list(event_days),
                        "population_filter_syntax": "Boolean SQL condition only, no leading WHERE",
                        "previous_execution_error": history[-1].get("tool_error") if history else None,
                        "concept": _concept_summary(selected),
                        "claim": selected["claim"],
                        "schema": tools.schema(), "created_features": tools.features,
                        "prior_experiments": [
                            {"question": t["test_question"],
                             "numerical_evidence": compact_query(t["result"])}
                            for t in selected["tests"][-3:]],
                        "scientist_request": decision["reason"],
                    }, "experiment-design-" + str(step), EXPERIMENT_SCHEMA)
                    measured_experiment = workbench.run(experiment_spec)
                    entry["experiment"] = measured_experiment
                    numeric = {"sql": measured_experiment["sql"],
                               "columns": measured_experiment["columns"],
                               "rows": measured_experiment["rows"]}
                    selected["tests"].append({
                        "version": selected["version"],
                        "test_question": experiment_spec["question"],
                        "predicted_falsifier": experiment_spec["falsifier"],
                        "result": numeric, "workbench": measured_experiment})
                    critique = critic.act({
                        "concept": _concept_summary(selected),
                        "hypothesis": selected["claim"],
                        "experiment_specification": experiment_spec,
                        "observed_statistics": measured_experiment["result"],
                        "negative_control": measured_experiment["negative_control"],
                        "request": "Evaluate rival mechanism, logging artifacts, and whether a DIFFERENT experiment is needed.",
                    }, "experiment-critique-" + str(step), CRITIC_SCHEMA)
                    entry["critique"] = critique
                    selected["critique"] = critique
                    if critique["decision"] == "test":
                        followup_spec = designer.act({
                            "concept": _concept_summary(selected),
                            "previous_experiment": experiment_spec,
                            "results": measured_experiment["result"],
                            "placebo": measured_experiment["negative_control"],
                            "critic_feedback": critique,
                            "schema": tools.schema(),
                            "available_event_day_range": list(event_days),
                            "instruction": "Propose a different experiment, grouping, control, time window or placebo that distinguishes the rival. Do not repeat the earlier specification.",
                        }, "experiment-countertest-" + str(step), EXPERIMENT_SCHEMA)
                        if followup_spec == experiment_spec:
                            entry["critic_experiment_error"] = "Repeated experiment specification"
                        else:
                            try:
                                followup = workbench.run(followup_spec)
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
                            and "critic_experiment" in entry
                            and selected["status"] == "investigating"):
                        reflector = ResearchAgent(model)
                        reflector.role = "discovery-reflection"
                        reflector.instruction = (
                            "You have an initial statistical experiment AND a countertest "
                            "requested by Critic. Their NUMERIC results are authoritative. "
                            "Decide whether to revise the mechanism/measurement, abandon "
                            "or request literature. Merely renaming the same hypothesis "
                            "is NOT revision. State a falsifiable alternative.")
                        reflection = reflector.act({
                            "concept": selected["claim"],
                            "initial_result": measured_experiment["result"],
                            "countertest_result": entry["critic_experiment"]["result"],
                            "critique": critique,
                        }, "experiment-reflection-" + str(step), REFLECTION_SCHEMA)
                        entry["reflection"] = reflection
                        revised = any(
                            reflection[key].strip() != selected["claim"][key].strip()
                            for key in ("mechanism", "rival", "falsifier", "measurement"))
                        if reflection["decision"] == "revise" and revised:
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
            history.append(entry)
            _save(dest / "journal.json", history)
            _save(dest / "research_state.json", {
                "concepts": concepts, "rounds_completed": len(history),
                "holdout_consulted": False})
        status = "frozen_unconfirmed" if last else "development_inconclusive"
        final = {"status": status, "frozen_id": last or None,
                 "concepts": concepts, "development_queries": tools.history,
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
