"""BADF-inspired evidence-first phenomenon extraction; no Bias acceptance here."""
import re

PHENOMENON_SCHEMA = {
    "type": "object", "properties": {"phenomena": {
        "type": "array", "maxItems": 4, "items": {
            "type": "object", "properties": {
                "description": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                "conditions": {"type": "string"},
                "observed_pattern": {"type": "string"},
                "alternative": {"type": "string"},
                "next_measurement": {"type": "string"}},
            "required": ["description", "evidence_ids", "conditions",
                         "observed_pattern", "alternative", "next_measurement"],
            "additionalProperties": False}}},
    "required": ["phenomena"], "additionalProperties": False,
}


def compact_profile(profile):
    return {"sampling": profile["sampling"], "fields": [
        {"source": table, "column": c["name"], "type": c["type"],
         "sample_distinct": c["sample_distinct"],
         "observed_levels": (c["observed_levels"][:2]
                             if c["sample_distinct"] <= 6 else [])}
        for table, section in profile["tables"].items()
        for c in section["columns"]]}


def evidence_for_agent(cards, queries, max_fields=8, max_contrasts=8):
    """Expose diverse actual contrasts, not only marginal frequency snippets."""
    def spread(items, n):
        if len(items) <= n:
            return items
        return [items[round(i*(len(items)-1)/(n-1))] for i in range(n)]
    fields = spread([c for c in cards if c["operation"]=="distribution_profile"],
                    max_fields)
    contrasts = spread([c for c in cards if c["operation"]=="generic_conditional_mean"],
                       max_contrasts)
    references = [c for c in cards
                  if c["operation"]=="matched_item_pool_reference"][:3]
    out = [{"id": c["observation_id"], "source": c["source_table"],
            "field": c["column"], "operation": c["operation"],
            "outcome": c.get("outcome"),
            "measured_direction": (f"mean({c['outcome']}) BY {c['column']}"
                                   if c.get("outcome") else "marginal distribution"),
            "sample_n": c["sample_n"],
            "values": c["measured_rows"][:4]} for c in fields + contrasts + references]
    out.extend({"id": "Q"+str(i+1).zfill(3), "query": q["sql"][:280],
                "rows": q["rows"][:5], "operation": "executed_dev_sql"}
               for i,q in enumerate(queries[-5:], start=max(0,len(queries)-5)))
    return out


def unify_phenomena(prior, proposals, evidence_catalog):
    """A claimed phenomenon needs relevant measured provenance.

    Marginal values cannot establish cross-table or conditional differences.
    No significance/causal claim is ever inferred by this extraction gate.
    """
    if not isinstance(evidence_catalog, dict):
        evidence_catalog = {k: {} for k in evidence_catalog}
    def key(row):
        return (re.sub(r"\s+", " ", row["description"].strip().casefold()),
                re.sub(r"\s+", " ", row["conditions"].strip().casefold()))
    known = {key(row): row for row in prior}
    for candidate in proposals:
        ids = sorted(set(candidate["evidence_ids"]) & set(evidence_catalog))
        if not ids or not candidate["description"].strip():
            continue
        cited = [evidence_catalog[x] for x in ids]
        statement = (" ".join(str(candidate.get(x,"")) for x in
                     ("description", "conditions", "observed_pattern"))).lower()
        # A cited field must appear in the claimed phenomenon; reject
        # spurious evidence links (e.g. older_click cited for recent_click).
        named = {e.get("field") for e in cited if e.get("field")}
        mentioned = {name for name in named if
                     re.search(r"(?<![a-z0-9_])"+re.escape(name.lower())+
                               r"(?![a-z0-9_])", statement)}
        if named and not named.issubset(mentioned):
            continue
        # Evidence records mean(outcome) BY group, never the reverse.
        # Require the claimed condition to name the actual grouping field.
        if any(e.get("operation") == "generic_conditional_mean" and
               e.get("field") and not re.search(
                   r"(?<![a-z0-9_])" + re.escape(e["field"].lower()) +
                   r"(?![a-z0-9_])", candidate["conditions"].lower())
               for e in cited):
            continue
        # Matched item pool is an eligible view of the events source, not
        # a separate table with independent sampling.
        sources = {("events" if e.get("source") == "shared_item_pool_events"
                    else e.get("source")) for e in cited if e.get("source")}
        if any(table in statement and table not in sources
               for table in ("events", "features")):
            continue
        relation = re.search(
            r"\b(compared|higher|lower|difference|differ|association|"
            r"correlation|versus|between)\b", statement)
        has_joint = any(e.get("operation") in
                        ("generic_conditional_mean", "executed_dev_sql",
                         "matched_item_pool_reference")
                        for e in cited)
        if relation and not has_joint:
            continue
        # Separate marginal samples from events/features are NOT matched
        # contexts, even if their schema contains the same field names.
        if len(sources)>1 and not any(
                e.get("operation")=="executed_dev_sql" for e in cited):
            continue
        # A Qwen sentence can reverse measured direction. Evidence values,
        # not verbal interpretations, own the numeric allocation statement.
        references = [e for e in cited
                      if e.get("operation") == "matched_item_pool_reference"]
        if references:
            ref = references[0]
            groups = ref.get("values", [])[:2]
            if groups:
                units = "; ".join(
                    f"{x['value']}: random share {x['random_share']}, "
                    f"standard share {x['standard_share']}, "
                    f"gap {x['share_gap']}, random click {x['random_click']}, "
                    f"standard click {x['standard_click']}"
                    for x in groups)
                candidate = {
                    **candidate,
                    "original_interpretation": candidate["observed_pattern"],
                    "observed_pattern": units + " (development-only descriptive)",
                    "description": "Observed matched-pool exposure contrast by "
                                   + ref["field"],
                    "conditions": "same video pool, dates and UI; grouping by "
                                  + ref["field"]}
        k = key(candidate)
        if k in known:
            known[k]["evidence_ids"] = sorted(set(known[k]["evidence_ids"]+ids))
            continue
        item = {**candidate, "id": "P"+str(len(prior)+1).zfill(3),
                "evidence_ids": ids, "status": "observed_unverified"}
        prior.append(item)
        known[k] = item
    return prior
