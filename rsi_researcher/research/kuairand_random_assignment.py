"""Random-insertion scientific probes for KuaiRand-27K.

Unlike observational history-condition cards, assignment of the *inserted
video* is randomized within the official insertion pool. Effects on later
recommendations are only identifiable as candidate intent-to-treat contrasts,
not a mechanism-specific policy/debias intervention.
"""
from __future__ import annotations

from pathlib import Path

def create_random_assignment_cards(db_path, min_events=80, limit=30,seed=20261024):
    import duckdb
    con=duckdb.connect(str(db_path))
    con.execute(f"""
        CREATE TABLE random_assignment_episodes AS
        SELECT uid,vid,event_day,moment_ms,tags AS assigned_tags,
               previous_tags,next_tags,next_click,next_long_view,click AS inserted_click,
               activity,tab,hash(concat(CAST(uid AS VARCHAR), ':', '{int(seed)}')) % 5 AS user_fold
        FROM observed_sequences
        WHERE is_random=1 AND next_is_random=0 AND next_tab=tab
              AND next_event_day=event_day
              AND next_moment-moment_ms BETWEEN 0 AND 7200000
              AND sequence_index>40
              AND tags IS NOT NULL AND next_tags IS NOT NULL
    """)
    # Future user folds are never used to select a candidate group.
    candidates=[r[0] for r in con.execute("""
        SELECT assigned_tags,COUNT(*) AS n
        FROM random_assignment_episodes
        WHERE user_fold<>0 AND event_day BETWEEN 20220422 AND 20220430
        GROUP BY assigned_tags HAVING n>=?
        ORDER BY n DESC LIMIT ?
    """,[min_events,limit]).fetchall()]
    cards=[]
    for label in candidates:
        outcomes=[]
        for name,col in (("future_same_tag","next_tags"),
                         ("pretreatment_same_tag","previous_tags")):
            for split,predicate in (
                ("development","user_fold<>0 AND event_day<=20220430"),
                ("confirmation","user_fold=0 AND event_day>=20220501")):
                n,nz,uz,treat,control=con.execute(f"""
                    SELECT COUNT(*) n,
                    COUNT(CASE WHEN assigned_tags=? THEN 1 END) nz,
                    COUNT(DISTINCT CASE WHEN assigned_tags=? THEN uid END) uz,
                    AVG(CASE WHEN assigned_tags=? THEN
                         CAST({col}=? AS INTEGER) END) treated,
                    AVG(CASE WHEN assigned_tags<>? THEN
                         CAST({col}=? AS INTEGER) END) control
                    FROM random_assignment_episodes WHERE {predicate}
                """,[label,label,label,label,label,label]).fetchone()
                if not n or nz<min_events or n-nz<min_events or uz<20:
                    continue
                if treat is None or control is None: continue
                outcomes.append({
                    "source_kind":"randomized_insertion_assignment",
                    "concept_id":"random_inserted_content:"+label,
                    "condition":"randomly assigned video has content code "+label,
                    "outcome":name,
                    "split":split,"n_total":n,"n_condition":nz,
                    "condition_users":uz,
                    "outcome_when_condition_true":round(treat,6),
                    "outcome_when_condition_false":round(control,6),
                    "observed_difference":round(treat-control,6),
                    "causal_claim_allowed":False,
                    "reference":"official randomized replacement video; does not randomize prior user state",
                    "meaning":"exploratory assignment contrast in next standard feed, not a verified mechanism or policy utility"
                })
        pre={r["split"]:r for r in outcomes if r["outcome"]=="pretreatment_same_tag"}
        post={r["split"]:r for r in outcomes if r["outcome"]=="future_same_tag"}
        if "development" not in post:continue
        cards.append({"development":{**post["development"],
                     "pretreatment_balance":(pre.get("development") or {}).get("observed_difference")},
                      "confirmation":({**post["confirmation"],
                     "pretreatment_balance":(pre.get("confirmation") or {}).get("observed_difference")}
                     if "confirmation" in post else None),
                      "is_causal_bias":False,
                      "experiment_needed":"verify pre-treatment tag balance, compare baseline item pool and real ranking state, test platform or replay state intervention"})
    con.close()
    return cards

def development_examples(db_path, cards, maximum=8):
    import duckdb
    con=duckdb.connect(str(db_path),read_only=True)
    evidence=[]
    for c in cards[:maximum]:
        label=c["development"]["concept_id"].split(":",1)[1]
        matches=con.execute("""
            SELECT uid,vid,event_day,moment_ms,assigned_tags,previous_tags,next_tags,
                   inserted_click,next_click,activity
            FROM random_assignment_episodes
            WHERE user_fold<>0 AND event_day<=20220430 AND assigned_tags=?
            ORDER BY hash(concat(CAST(uid AS VARCHAR),':',CAST(moment_ms AS VARCHAR)))
            LIMIT 4
        """,[label]).fetchall()
        fields=["uid","vid","event_day","moment_ms","assigned_tags",
                "previous_tags","next_tags","inserted_click","next_click","activity"]
        evidence.append({"concept_id":c["development"]["concept_id"],
                         "outcome":c["development"]["outcome"],
                         "raw_event_keys_and_context":[dict(zip(fields,row)) for row in matches]})
    con.close()
    return evidence
