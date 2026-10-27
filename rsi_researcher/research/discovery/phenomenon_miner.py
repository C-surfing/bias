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
         "observed_levels": c["observed_levels"][:3]}
        for table, section in profile["tables"].items()
        for c in section["columns"]]}


def evidence_for_agent(cards, queries, max_fields=12, max_contrasts=12):
    """Expose diverse actual contrasts, not only marginal frequency snippets."""
    def spread(items, n):
        if len(items) <= n:
            return items
        return [items[round(i*(len(items)-1)/(n-1))] for i in range(n)]
    fields = spread([c for c in cards if c["operation"]=="distribution_profile"],
                    max_fields)
    contrasts = spread([c for c in cards if c["operation"]=="generic_conditional_mean"],
                       max_contrasts)
    out = [{"id": c["observation_id"], "source": c["source_table"],
            "field": c["column"], "operation": c["operation"],
            "outcome": c.get("outcome"), "sample_n": c["sample_n"],
            "values": c["measured_rows"][:6]} for c in fields + contrasts]
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
        sources = {e.get("source") for e in cited if e.get("source")}
        if any(table in statement and table not in sources
               for table in ("events", "features")):
            continue
        relation = re.search(
            r"\b(compared|higher|lower|difference|differ|association|"
            r"correlation|versus|between)\b", statement)
        has_joint = any(e.get("operation") in
                        ("generic_conditional_mean", "executed_dev_sql")
                        for e in cited)
        if relation and not has_joint:
            continue
        # Separate marginal samples from events/features are NOT matched
        # contexts, even if their schema contains the same field names.
        if len(sources)>1 and not any(
                e.get("operation")=="executed_dev_sql" for e in cited):
            continue
        k = key(candidate)
        if k in known:
            known[k]["evidence_ids"] = sorted(set(known[k]["evidence_ids"]+ids))
            continue
        item = {**candidate, "id": "P"+str(len(prior)+1).zfill(3),
                "evidence_ids": ids, "status": "observed_unverified"}
        prior.append(item)
        known[k] = item
    return prior
