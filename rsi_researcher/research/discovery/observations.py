"""Development-only observations: executable evidence, never named bias candidates.

Cards document what was measured and preserve their SQL; observations do not
establish item merit, policy causality, or a normative exposure target.
"""
from __future__ import annotations


def build_observation_cards(tools) -> list[dict]:
    """Bounded, diverse, reproducible evidence for Qwen's first research decisions."""
    schema = tools.schema()
    events = {x["column"] for x in schema["events"]}
    features = {x["column"] for x in schema["features"]}
    cards = []

    def observe(identifier, question, table, required, sql, limit, caveat):
        columns = events if table == "events" else features
        if not set(required).issubset(columns):
            return
        result = tools.connection.execute(sql)
        names = [x[0] for x in result.description]
        rows = [dict(zip(names, row)) for row in result.fetchmany(limit)]
        cards.append({
            "observation_id": identifier,
            "question": question,
            "source_table": table,
            "measurement_sql": sql.strip(),
            "measured_rows": rows,
            "evidence_level": "descriptive_development_only",
            "caveat": caveat,
        })

    # Standard and randomly inserted exposures coexist only on overlapping
    # dates. A difference between them is NOT an algorithmic-bias estimand.
    common = ("event_day IN (SELECT event_day FROM events GROUP BY event_day "
              "HAVING COUNT(*) FILTER (WHERE is_random=0)>0 "
              "AND COUNT(*) FILTER (WHERE is_random=1)>0)")
    observe(
        "O001", "What differs between exposure modes on shared dates?",
        "events", ("event_day", "is_random", "uid", "vid", "click"),
        "SELECT is_random,COUNT(*) n_events,COUNT(DISTINCT uid) n_users,"
        "COUNT(DISTINCT vid) n_items,ROUND(AVG(click),5) click_rate "
        "FROM events WHERE " + common + " GROUP BY 1 ORDER BY 1",
        2, "Modes draw from different item policies and opportunity sets; no causal or merit claim.")
    observe(
        "O002", "Which observed item groups have support in both modes?",
        "events", ("event_day", "is_random", "tags", "click"),
        "SELECT tags,COUNT(*) FILTER (WHERE is_random=0) n_standard,"
        "COUNT(*) FILTER (WHERE is_random=1) n_random,"
        "ROUND(AVG(click) FILTER (WHERE is_random=0),5) response_standard,"
        "ROUND(AVG(click) FILTER (WHERE is_random=1),5) response_random "
        "FROM events WHERE " + common + " AND tags IS NOT NULL "
        "GROUP BY tags HAVING COUNT(*) FILTER (WHERE is_random=1)>=100 "
        "AND COUNT(*) FILTER (WHERE is_random=0)>=100 "
        "ORDER BY n_random DESC LIMIT 12",
        12, "Tag values are opaque IDs and may encode multiple tags; response is not merit.")
    observe(
        "O003", "Does previous feedback interact with exposure mode?",
        "events", ("is_random", "previous_click", "uid", "click"),
        "SELECT is_random,previous_click,COUNT(*) n_events,"
        "COUNT(DISTINCT uid) n_users,ROUND(AVG(click),5) response "
        "FROM events WHERE " + common + " AND previous_click IN (0,1) "
        "GROUP BY 1,2 ORDER BY 1,2",
        4, "Previous-click differences can reflect user interest or sequential logging.")
    observe(
        "O004", "Are the exposure populations and responses stable over time?",
        "events", ("event_day", "is_random", "uid", "click"),
        "SELECT event_day,is_random,COUNT(*) n_events,"
        "COUNT(DISTINCT uid) n_users,ROUND(AVG(click),5) response "
        "FROM events WHERE " + common +
        " GROUP BY 1,2 ORDER BY event_day DESC,is_random LIMIT 24",
        24, "Date shifts and selection can confound all cross-mode contrasts.")
    observe(
        "O005", "How do pre-event context and observed response vary by users?",
        "features", ("activity", "after_long_gap", "uid", "click"),
        "SELECT activity,after_long_gap,COUNT(*) n_events,"
        "COUNT(DISTINCT uid) n_users,ROUND(AVG(click),5) response "
        "FROM features WHERE after_long_gap IN (0,1) "
        "GROUP BY 1,2 ORDER BY n_events DESC LIMIT 12",
        12, "This only describes response variation; it is not item exposure distortion.")
    observe(
        "O006", "Which authors of recommended items have support in both modes?",
        "events", ("event_day", "is_random", "author", "click"),
        "SELECT author,COUNT(*) FILTER (WHERE is_random=0) n_standard,"
        "COUNT(*) FILTER (WHERE is_random=1) n_random,"
        "ROUND(AVG(click) FILTER (WHERE is_random=0),5) response_standard,"
        "ROUND(AVG(click) FILTER (WHERE is_random=1),5) response_random "
        "FROM events WHERE " + common + " AND author IS NOT NULL "
        "GROUP BY author HAVING COUNT(*) FILTER (WHERE is_random=0)>=100 "
        "AND COUNT(*) FILTER (WHERE is_random=1)>=20 "
        "ORDER BY n_random DESC LIMIT 12", 12,
        "Author allocation is descriptive; author-specific supply and merit remain unknown.")
    observe(
        "O007", "Does interface placement context alter observed exposure composition?",
        "events", ("event_day", "is_random", "tab", "uid", "click"),
        "SELECT is_random,tab,COUNT(*) n_events,COUNT(DISTINCT uid) n_users,"
        "ROUND(AVG(click),5) response "
        "FROM events WHERE " + common + " AND tab IS NOT NULL "
        "GROUP BY 1,2 ORDER BY n_events DESC LIMIT 12", 12,
        "The tab label describes a collection context, not a randomized assignment.")
    return cards
