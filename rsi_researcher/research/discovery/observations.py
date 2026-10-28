"""Schema-derived, non-prescriptive evidence for autonomous discovery."""
from collections import Counter
import math


def build_data_profile(tools, sample_size=4000):
    profile = {"sampling": "DuckDB reservoir seed=17", "tables": {}}
    for table in ("events", "features", *getattr(tools, "reference_views", [])):
        query = f"SELECT * FROM {table} USING SAMPLE {sample_size} ROWS (reservoir, 17)"
        cursor = tools.connection.execute(query)
        names = [r[0] for r in cursor.description]
        rows = cursor.fetchall()
        columns = []
        for i, name in enumerate(names):
            values = [r[i] for r in rows if r[i] is not None]
            frequency = Counter(str(v) for v in values)
            top = sorted(frequency.items(), key=lambda p: (-p[1], p[0]))[:8]
            columns.append({
                "name": name, "type": tools.schema()[table][i]["type"],
                "sample_distinct": len(frequency),
                "sample_non_null": len(values),
                "observed_levels": [{"value": v, "count": n} for v,n in top],
                "sample_null": len(rows) - len(values)})
        profile["tables"][table] = {
            "sample_rows": len(rows), "columns": columns, "source_sql": query}
    return profile


def build_exploration_evidence(tools, profile=None):
    """Compatibility name: cards are generated per observed FIELD, never topic."""
    profile = profile if profile is not None else build_data_profile(tools)
    cards = []
    for table, part in profile["tables"].items():
        for c in part["columns"]:
            cards.append({
                "observation_id": "E" + str(len(cards)+1).zfill(3),
                "operation": "distribution_profile",
                "column": c["name"], "source_table": table,
                "question": "Observed values in " + table + "." + c["name"],
                "measurement_sql": part["source_sql"],
                "measured_rows": c["observed_levels"],
                "sample_n": part["sample_rows"],
                "evidence_level": "sampled_descriptive_development_only",
                "caveat": "Sampled field distribution, not causal or bias evidence."})
    # Data-driven conditional scans: enumerate every eligible pair from the
    # observed schema and spread a small sample across that space. No preset
    # exposure/response pair, hypothesis or significance ranking is used.
    from collections import defaultdict
    native = supported_native_levels(profile)
    pairs = []
    for table, section in profile["tables"].items():
        outcomes = [c["name"] for c in section["columns"]
                    if c["type"].upper().startswith(
                        ("INTEGER", "INT", "BIGINT", "DOUBLE", "FLOAT", "SMALLINT"))
                    and 2 <= c["sample_distinct"] <= 20
                    and c["name"] not in ("uid", "vid", "event_day", "moment_ms")]
        for source, group in native:
            if source == table:
                pairs.extend((table, group, y) for y in outcomes if y != group)
    profile["eligible_conditional_pairs"] = len(pairs)
    if pairs:
        amount = min(len(pairs), 24)
        indices = list(dict.fromkeys(
            round(i * (len(pairs)-1) / max(1, amount-1))
            for i in range(amount)))
        cached = {}
        for table in profile["tables"]:
            cursor = tools.connection.execute(
                f"SELECT * FROM {table} USING SAMPLE 6000 ROWS (reservoir, 23)")
            cached[table] = ([d[0] for d in cursor.description], cursor.fetchall())
        for index in indices:
            table, group, outcome = pairs[index]
            columns, sampled = cached[table]
            g, y = columns.index(group), columns.index(outcome)
            aggregates = defaultdict(list)
            for row in sampled:
                if row[g] is not None and isinstance(row[y], (int, float)):
                    aggregates[str(row[g])].append(float(row[y]))
            rows = [{"group_value": key, "n": len(values),
                     "mean": round(sum(values) / len(values), 6)}
                    for key, values in sorted(aggregates.items())
                    if len(values) >= 5]
            if len(rows) < 2:
                continue
            cards.append({
                "observation_id": "E" + str(len(cards)+1).zfill(3),
                "operation": "generic_conditional_mean",
                "source_table": table, "column": group, "outcome": outcome,
                "question": f"Sampled {outcome} conditional on {group}",
                "measurement_sql": f"reservoir_group_mean({table},{group},{outcome})",
                "measured_rows": rows[:8], "sample_n": len(sampled),
                "evidence_level": "sampled_descriptive_development_only",
                "caveat": "Exploratory sample contrast; no independence, causal or bias identification."})
    for ref in build_matched_reference_cards(tools):
        ref["observation_id"] = "E" + str(len(cards)+1).zfill(3)
        cards.append(ref)
    return cards


def build_matched_reference_cards(tools):
    """Schema-wide same-pool reference support, no preset bias hypotheses."""
    if "shared_item_pool_events" not in getattr(tools, "reference_views", []):
        return []
    source = "shared_item_pool_events"
    candidates = [c["column"] for c in tools.schema()[source]
                  if c["type"].upper().startswith(("VARCHAR", "TEXT"))]
    totals = dict(tools.connection.execute(
        "SELECT is_random,COUNT(*) FROM " + source + " GROUP BY 1").fetchall())
    nr, ns = totals.get(1, 0), totals.get(0, 0)
    if min(nr, ns) == 0:
        return []
    out = []
    for field in candidates:
        query = (f"SELECT {field}, "
                 "COUNT(*) FILTER(WHERE is_random=1) nr, "
                 "COUNT(*) FILTER(WHERE is_random=0) ns, "
                 "AVG(click::DOUBLE) FILTER(WHERE is_random=1) rc, "
                 "AVG(click::DOUBLE) FILTER(WHERE is_random=0) sc, "
                 "COUNT(DISTINCT uid) FILTER(WHERE is_random=1) ru, "
                 "COUNT(DISTINCT uid) FILTER(WHERE is_random=0) su "
                 f"FROM {source} WHERE {field} IS NOT NULL GROUP BY 1")
        data = tools.connection.execute(query).fetchall()
        rows = [
            {"value": str(v), "random_n": int(r), "standard_n": int(t),
             "random_share": round(r/nr, 5), "standard_share": round(t/ns, 5),
             "share_gap": round(t/ns-r/nr, 5),
             "random_click": round(float(rc), 5),
             "standard_click": round(float(sc), 5),
             "random_users": int(ru), "standard_users": int(su)}
            for v,r,t,rc,sc,ru,su in data
            if v is not None and r >= 60 and t >= 30
            and rc is not None and sc is not None]
        if len(rows) < 2:
            continue
        rows.sort(key=lambda x: -abs(x["share_gap"]))
        out.append({"operation": "matched_item_pool_reference",
                    "source_table": source, "column": field,
                    "question": "Same-pool allocation and response by " + field,
                    "measurement_sql": query, "measured_rows": rows[:8],
                    "sample_n": int(nr+ns),
                    "evidence_level": "development_reference_support_only",
                    "caveat": "Random insertion item pool and matched period/UI. "
                    "Raw allocation gap does not identify recommender-caused "
                    "harm or preference without user-level controls."})
    return out


def supported_native_levels(profile):
    """Ground native candidate groups in measured values, not handpicked cards."""
    excluded = {"uid", "vid", "event_day", "moment_ms",
                "click", "hate", "long_view", "next_click"}
    found = {}
    for table, section in profile["tables"].items():
        for c in section["columns"]:
            if c["name"] in excluded or not 2 <= c["sample_distinct"] <= 6:
                continue
            levels = c["observed_levels"]
            if (len(levels) == c["sample_distinct"]
                    and all(p["count"] >= 5 for p in levels)):
                found[(table,c["name"])] = [v["value"] for v in levels]
    return found
