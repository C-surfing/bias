"""KuaiRand-27K open-ended sequential anomaly discovery, using complete logged histories.

Scientific contract: this module creates observational anomaly and scientific
hypothesis evidence. It never upgrades observed correlations to causal Bias.
KuaiRand-Pure is expressly rejected: its filtered histories are incomplete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

STANDARD_PARTS = (
    "log_standard_4_08_to_4_21_27k_part1.csv",
    "log_standard_4_08_to_4_21_27k_part2.csv",
    "log_standard_4_22_to_5_08_27k_part1.csv",
    "log_standard_4_22_to_5_08_27k_part2.csv",
)
RANDOM_LOG = "log_random_4_22_to_5_08_27k.csv"
BASIC = "video_features_basic_27k.csv"
USERS = "user_features_27k.csv"

def discover_files(directory):
    base=Path(directory)
    candidates=(base/"data",base/"KuaiRand-27K"/"data",base)
    for folder in candidates:
        if all((folder/n).is_file() for n in (*STANDARD_PARTS,RANDOM_LOG,BASIC,USERS)):
            return folder
    raise FileNotFoundError(
        "Complete KuaiRand-27K is required (four full standard logs, random log, "
        "video basic metadata and users). Pure cannot support temporal discovery.")

def _sqlstr(value):
    return "'" + str(value).replace("'","''") + "'"

def _query_values(paths):
    return "[" + ",".join(_sqlstr(p) for p in paths) + "]"

def prepare_trajectories(root, database, user_modulo=32, threads=8, seed=20261023):
    import duckdb
    folder=discover_files(root)
    db=Path(database)
    db.parent.mkdir(parents=True,exist_ok=True)
    if db.exists(): raise FileExistsError("Never silently overwrite an experiment database")
    connection=duckdb.connect(str(db))
    connection.execute(f"SET threads={int(threads)}")
    connection.execute("SET memory_limit='28GB'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET temp_directory="+_sqlstr(str(db.parent/"spill")))
    sources=_query_values([folder/name for name in (*STANDARD_PARTS,RANDOM_LOG)])
    # Keep immutable source column semantics; later model input never includes
    # confirmation cohort outcomes.
    connection.execute(f"""
        CREATE TABLE events AS SELECT
          regexp_extract(filename,'[^/]+$') AS source_file,
          TRY_CAST(user_id AS INTEGER) uid,
          TRY_CAST(video_id AS BIGINT) vid,
          TRY_CAST(date AS INTEGER) event_day,
          TRY_CAST(time_ms AS BIGINT) moment_ms,
          TRY_CAST(is_click AS INTEGER) click,
          TRY_CAST(is_hate AS INTEGER) hate,
          TRY_CAST(is_like AS INTEGER) liked,
          TRY_CAST(is_rand AS INTEGER) is_random,
          TRY_CAST(long_view AS INTEGER) long_view,
          TRY_CAST(tab AS VARCHAR) tab
        FROM read_csv({sources},header=true,all_varchar=true,union_by_name=true,filename=true)
        WHERE TRY_CAST(user_id AS INTEGER) IS NOT NULL
          AND hash(concat(user_id, ':', '{int(seed)}')) % {int(user_modulo)} = 0
          AND is_rand IN ('0','1')
          AND TRY_CAST(date AS INTEGER) BETWEEN 20220408 AND 20220508
          AND TRY_CAST(time_ms AS BIGINT) IS NOT NULL
    """)
    # Same item may appear in repeated records. Reading basic metadata is
    # independent of reward outcomes, hence it cannot leak click labels.
    connection.execute(f"""
        CREATE TABLE metadata AS SELECT TRY_CAST(video_id AS BIGINT) vid,
        NULLIF(tag,'') AS tags, author_id AS author,
        TRY_CAST(video_duration AS DOUBLE) duration_ms
        FROM read_csv({_sqlstr(folder/BASIC)},header=true,all_varchar=true)
        WHERE TRY_CAST(video_id AS BIGINT) IN (SELECT DISTINCT vid FROM events)
    """)
    connection.execute(f"""
        CREATE TABLE users AS SELECT TRY_CAST(user_id AS INTEGER) uid,
          user_active_degree AS activity
        FROM read_csv({_sqlstr(folder/USERS)},header=true,all_varchar=true)
    """)
    # Only PRIOR activity: window does not include the current event's reward.
    # Group by user and explicit millisecond sequence to retain full history.
    connection.execute("""
        CREATE TABLE observed_sequences AS
        SELECT e.uid,e.vid,e.event_day,e.moment_ms,e.click,e.hate,e.long_view,
               e.is_random,e.source_file,
               e.tab,coalesce(u.activity,'unknown') activity,
               m.author,m.tags,
               row_number() OVER w AS sequence_index,
               lag(e.vid,1) OVER w AS previous_video,
               lag(m.tags,1) OVER w AS previous_tags,
               lag(e.moment_ms,1) OVER w AS previous_moment,
               lag(e.click,1) OVER w AS previous_click,
               lag(e.hate,1) OVER w AS previous_hate,
               lag(e.long_view,1) OVER w AS previous_long_view,
               lag(e.click,2) OVER w AS click_two_steps_back,
               lead(e.click,1) OVER w AS next_click,
               lead(m.tags,1) OVER w AS next_tags,
               lead(e.tab,1) OVER w AS next_tab,
               lead(e.is_random,1) OVER w AS next_is_random,
               lead(e.long_view,1) OVER w AS next_long_view,
               lead(e.event_day,1) OVER w AS next_event_day,
               lead(e.moment_ms,1) OVER w AS next_moment,
               avg(e.click) OVER
                 (PARTITION BY e.uid ORDER BY e.moment_ms,e.vid
                  ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING) AS recent_click,
               avg(e.click) OVER
                 (PARTITION BY e.uid ORDER BY e.moment_ms,e.vid
                  ROWS BETWEEN 40 PRECEDING AND 9 PRECEDING) AS older_click
        FROM events e
        LEFT JOIN metadata m ON e.vid=m.vid
        LEFT JOIN users u ON e.uid=u.uid
        WINDOW w AS (PARTITION BY e.uid ORDER BY e.moment_ms,e.vid)
    """)
    # Previous-day aggregate excludes this user's own observations before
    # it is called cross-user evidence. Still observational, not influence.
    connection.execute("""
        CREATE TABLE item_day AS
        SELECT vid,event_day,COUNT(*) n,SUM(hate) hates
        FROM events GROUP BY vid,event_day
    """)
    connection.execute("""
        CREATE TABLE own_item_day AS
        SELECT uid,vid,event_day,COUNT(*) n,SUM(hate) hates
        FROM events GROUP BY uid,vid,event_day
    """)
    connection.execute(f"""
        CREATE TABLE all_anomaly_features AS
        SELECT s.uid,s.vid,s.event_day,s.moment_ms,s.source_file,s.is_random,
               s.activity,s.tags,s.click,s.long_view,
               CASE WHEN s.next_event_day=s.event_day
                          AND s.next_moment-s.moment_ms BETWEEN 0 AND 7200000
                    THEN s.next_click ELSE NULL END AS next_click,
               hash(concat(CAST(s.uid AS VARCHAR), ':', '{int(seed)}')) % 5 AS user_fold,
               CASE WHEN s.moment_ms-s.previous_moment > 7200000
                     THEN 1 ELSE 0 END AS after_long_gap,
               CASE WHEN s.previous_click=0 THEN 1 ELSE 0 END AS after_nonclick,
               CASE WHEN s.previous_hate=1 THEN 1 ELSE 0 END AS after_negative_feedback,
               CASE WHEN s.vid=s.previous_video THEN 1 ELSE 0 END AS immediate_repeat,
               CASE WHEN s.tags IS NOT NULL AND s.previous_tags IS NOT NULL
                          AND s.tags <> s.previous_tags THEN 1 ELSE 0 END AS topic_changed,
               CASE WHEN s.recent_click IS NOT NULL AND s.older_click IS NOT NULL
                     AND s.recent_click+0.10<s.older_click THEN 1 ELSE 0 END AS recent_response_drop,
               CASE WHEN i.n-COALESCE(o.n,0) >= 3
                        AND (i.hates-COALESCE(o.hates,0))*1.0
                            /NULLIF(i.n-COALESCE(o.n,0),0)>0.12
                     THEN 1 ELSE 0 END
                     AS after_other_users_negative_response,
               CAST(s.moment_ms-s.previous_moment AS BIGINT) AS previous_interval_ms,
               s.recent_click,s.older_click
        FROM observed_sequences s
        LEFT JOIN item_day i ON i.vid=s.vid
          AND strptime(CAST(i.event_day AS VARCHAR),'%Y%m%d')
               = strptime(CAST(s.event_day AS VARCHAR),'%Y%m%d')-INTERVAL '1 DAY'
        LEFT JOIN own_item_day o ON o.uid=s.uid AND o.vid=s.vid
          AND o.event_day=i.event_day
        WHERE s.sequence_index>40
    """)
    connection.execute("CREATE VIEW anomaly_features AS SELECT * FROM all_anomaly_features WHERE is_random=0")
    connection.execute("CREATE VIEW random_probe_features AS SELECT * FROM all_anomaly_features WHERE is_random=1")
    stats={}
    for relation in ('events','metadata','users','observed_sequences','anomaly_features',
                     'random_probe_features'):
        stats[relation]=connection.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]
    connection.close()
    return {"database":str(db),"source":"KuaiRand-27K complete standard logs",
            "source_files":[str(folder/n) for n in (*STANDARD_PARTS,RANDOM_LOG)],
            "user_sampling":f"seeded DuckDB hash(user_id:seed={seed}) modulo {user_modulo}",
            "tables":stats}

FEATURES=(
    "after_long_gap","after_nonclick","after_negative_feedback",
    "immediate_repeat","topic_changed","recent_response_drop",
    "after_other_users_negative_response",
)
OUTCOMES=("click","long_view","next_click")

def mine_observational_cards(db_path,seed=20261023,min_events=180,min_users=30):
    """Discovery on 80% of users; confirm separately on future 20% users."""
    import duckdb
    con=duckdb.connect(str(db_path),read_only=True)
    # Pairwise interactions are generated from observed trajectories and
    # selected by residual outcome structure, not by Bias category names.
    # A condition is a *slice*: no treatment effect is inferred here.
    expressions=[(f, f"{f}=1") for f in FEATURES]
    expressions += [(f"{a}&{b}",f"{a}=1 AND {b}=1")
                    for idx,a in enumerate(FEATURES) for b in FEATURES[idx+1:]]
    cards=[]
    for metric in OUTCOMES:
        for concept,condition in expressions:
            # All observations (including controls) are from *disjoint users*
            # in development and confirmation, and different calendar windows.
            for split,predicate in (
                ("development","user_fold<>0 AND event_day<=20220430"),
                ("confirmation","user_fold=0 AND event_day>=20220501"),
            ):
                q=f"""
                    SELECT COUNT(*) n,COUNT(DISTINCT uid) users,
                      SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) n_target,
                      COUNT(DISTINCT CASE WHEN {condition} THEN uid END) users_target,
                      AVG(CASE WHEN {condition} THEN {metric} END) target_rate,
                      AVG(CASE WHEN NOT ({condition}) THEN {metric} END) comparison_rate
                    FROM anomaly_features WHERE {predicate} AND {metric} IS NOT NULL
                """
                n,users,n_tar,u_tar,p,q0=con.execute(q).fetchone()
                if (n_tar is None or n_tar<min_events or
                    u_tar<min_users or n-n_tar<min_events or
                    p is None or q0 is None): continue
                diff=p-q0
                se=(max(0,p*(1-p))/n_tar+
                    max(0,q0*(1-q0))/(n-n_tar))**0.5
                evidence={
                    "concept_id":concept,"condition":condition,"outcome":metric,
                    "split":split,"n_total":n,"n_condition":n_tar,
                    "unique_users":users,"condition_users":u_tar,
                    "outcome_when_condition_true":round(p,6),
                    "outcome_when_condition_false":round(q0,6),
                    "observed_difference":round(diff,6),
                    "naive_event_level_ci95":[round(diff-1.96*se,6),round(diff+1.96*se,6)],
                    "causal_claim_allowed":False}
                if "&" in concept:
                    a,b=concept.split("&")
                    # The interaction is a departure from additive conditional
                    # response, NOT the naive group 11 vs all remaining users.
                    cell_rows=con.execute(f"""
                        SELECT {a} a, {b} b, COUNT(*) n,
                               COUNT(DISTINCT uid) users,AVG({metric}) response
                        FROM anomaly_features
                        WHERE {predicate} AND {metric} IS NOT NULL
                        GROUP BY {a},{b}
                    """).fetchall()
                    cells={(int(a0),int(b0)):(n0,u0,y0)
                           for a0,b0,n0,u0,y0 in cell_rows
                           if a0 is not None and b0 is not None}
                    if len(cells)!=4 or any(v[0]<min_events or
                                            v[1]<min_users
                                            for v in cells.values()):
                        continue
                    y11=cells[(1,1)][2]
                    y10=cells[(1,0)][2]
                    y01=cells[(0,1)][2]
                    y00=cells[(0,0)][2]
                    residual=y11-y10-y01+y00
                    evidence["factorial_interaction_residual"]=round(residual,6)
                    evidence["four_condition_cells"]={
                        f"{i}{j}":{"events":cells[(i,j)][0],
                                    "users":cells[(i,j)][1],
                                    "outcome_mean":round(cells[(i,j)][2],6)}
                        for i in (0,1) for j in (0,1)}
                    evidence["interpretation"]="observational difference-in-differences of conditions; NOT a causal interaction"
                cards.append(evidence)
    con.close()
    dev=[c for c in cards if c["split"]=="development"]
    hold={(c["concept_id"],c["outcome"]):c
          for c in cards if c["split"]=="confirmation"}
    # Deterministic candidate pre-ranking by descriptive strength and support,
    # never by hidden confirmation labels.
    # Evaluate conditional interactions by their *incremental* residual,
    # not an accidental strong first-order effect of one constituent.
    dev.sort(key=lambda c:-(abs(c.get("factorial_interaction_residual",
                                     c["observed_difference"])) *
                            min(1,c["n_condition"]/1000)))
    return [{"development":c,"confirmation":hold.get((c["concept_id"],c["outcome"])),
             "is_causal_bias":False,
             "experiment_needed":"control item quality, user state and policy; test randomized or executable intervention"}
            for c in dev]

def development_trajectory_samples(db_path, cards, maximum=8):
    """Public, deterministic raw-event evidence; never touches confirmation."""
    import duckdb
    db=duckdb.connect(str(db_path),read_only=True)
    examples=[]
    for row in cards[:maximum]:
        condition=row["development"]["condition"]
        records=db.execute(f"""
            SELECT uid,vid,event_day,moment_ms,source_file,tags,activity,click,long_view,
                   recent_click,older_click,after_nonclick,
                   after_negative_feedback,after_long_gap,
                   after_other_users_negative_response
            FROM anomaly_features
            WHERE user_fold<>0 AND event_day<=20220430 AND ({condition})
            ORDER BY hash(concat(CAST(uid AS VARCHAR),':',CAST(moment_ms AS VARCHAR)))
            LIMIT 4
        """).fetchall()
        keys=["uid","vid","event_day","moment_ms","source_file","tags","activity","click",
              "long_view","recent_click","older_click","after_nonclick",
              "after_negative_feedback","after_long_gap",
              "after_other_users_negative_response"]
        examples.append({"concept_id":row["development"]["concept_id"],
                         "outcome":row["development"]["outcome"],
                         "raw_event_keys_and_context":[dict(zip(keys,r)) for r in records]})
    db.close()
    return examples

def random_reference_audit(db_path, development):
    """Independent randomized-insertion response within same observed context.

    This estimates conditional *feedback* on the randomized item pool, not
    the welfare effect of changing a recommender action. No IPS extrapolation.
    """
    import duckdb,math
    con=duckdb.connect(str(db_path),read_only=True)
    condition=development["condition"]
    metric=development["outcome"]
    if metric=="next_click":
        return {"status":"not_identified","reason":
                "The next recommendation is not randomized by the current random insertion"}
    raw=con.execute(f"""
        SELECT count(*) total,COUNT(DISTINCT uid) users,
          COUNT(CASE WHEN ({condition}) THEN 1 END) n_condition,
          COUNT(DISTINCT CASE WHEN ({condition}) THEN uid END) users_condition,
          AVG(CASE WHEN ({condition}) THEN {metric} END) conditional_feedback,
          AVG(CASE WHEN NOT ({condition}) THEN {metric} END) otherwise_feedback
        FROM random_probe_features
        WHERE user_fold=0 AND event_day>=20220501 AND {metric} IS NOT NULL
    """).fetchone()
    con.close()
    total,users,n_target,units_target,p,q=raw
    if total<300 or n_target<80 or total-n_target<80 or units_target<30:
        return {"status":"insufficient_randomized_support",
                "n_random":total,"n_condition":n_target,
                "n_unique_users":users,"n_condition_users":units_target,
                "utility_identified":False}
    effect=p-q
    se=math.sqrt(max(0,p*(1-p))/n_target+
                 max(0,q*(1-q))/(total-n_target))
    return {"status":"randomized_insertion_descriptive_reference",
            "n_random":total,"n_condition":n_target,"n_unique_users":users,
            "conditional_feedback":round(p,6),
            "otherwise_feedback":round(q,6),
            "conditional_minus_other":round(effect,6),
            "naive_event_level_ci95":[round(effect-1.96*se,6),
                                      round(effect+1.96*se,6)],
            "random_intervention":"item replacement only, not the user's history state",
            "causal_debias_utility_identified":False,
            "limitations":["random insertion restricted to ~7583 candidate videos",
                           "history condition was not randomly assigned",
                           "impression-level interval ignores repeated users and policy placement"]}

def matched_independent_review(db_path, development):
    """Only AFTER a Qwen claim is frozen: user/content-stratified heldout check.

    Matched observational outcome: never described as randomized ranking
    intervention, user preference truth, or mechanism-targeted Debias.
    """
    import duckdb
    condition=development["condition"]
    outcome=development["outcome"]
    db=duckdb.connect(str(db_path),read_only=True)
    # Both condition states must occur *within a user and a video tag stratum*.
    # User-level averages, not impression-level pseudo-independent p-values.
    rows=db.execute(f"""
        WITH matched AS (
          SELECT uid, COALESCE(tags,'unknown') AS content_tags,
                 AVG(CASE WHEN ({condition}) THEN {outcome} END) active_mean,
                 AVG(CASE WHEN NOT ({condition}) THEN {outcome} END) control_mean,
                 COUNT(CASE WHEN ({condition}) THEN 1 END) active_n,
                 COUNT(CASE WHEN NOT ({condition}) THEN 1 END) control_n
          FROM anomaly_features
          WHERE user_fold=0 AND event_day>=20220501
            AND {outcome} IS NOT NULL
          GROUP BY uid, COALESCE(tags,'unknown')
          HAVING active_n>=2 AND control_n>=2
        )
        SELECT uid, SUM((active_mean-control_mean)*LEAST(active_n,control_n))
                        /SUM(LEAST(active_n,control_n)) AS delta,
               SUM(LEAST(active_n,control_n)) AS overlap_n
        FROM matched GROUP BY uid
    """).fetchall()
    db.close()
    import math
    values=[float(r[1]) for r in rows if r[1] is not None]
    if len(values)<30:
        return {"status":"not_identified","reason":"too few same-user/tag matched units",
                "matched_users":len(values),"causal_identification":False}
    effect=sum(values)/len(values)
    spread=sum((x-effect)**2 for x in values)/(len(values)-1)
    se=(spread/len(values))**.5
    lo=effect-1.96*se
    hi=effect+1.96*se
    return {"status":"matched_observational_only","matched_users":len(values),
            "user_average_difference":round(effect,6),
            "user_cluster_ci95":[round(lo,6),round(hi,6)],
            "dev_holdout_direction_consistent":effect*development["observed_difference"]>0,
            "excludes_zero":lo>0 or hi<0,
            "causal_identification":False,
            "utility_harm_identified":False,
            "debias_verified":False,
            "limitations":["user/item-tag matching does not remove changing preferences",
                           "candidate pool, feedback position and time-varying content quality remain",
                           "exploratory post-selection intervals not multiplicity corrected"]}

def run(root,output,modulo=32,seed=20261023,min_events=180):
    dest=Path(output)
    dest.mkdir(parents=True,exist_ok=False)
    db=dest/"seq.duckdb"
    build=prepare_trajectories(root,db,modulo,seed=seed)
    observational=mine_observational_cards(db,seed=seed,min_events=min_events)
    from rsi_researcher.research.kuairand_random_assignment import (
        create_random_assignment_cards,development_examples)
    randomized=create_random_assignment_cards(db,seed=seed,min_events=max(80,min_events//2))
    # Assignment shocks have a more promising empirical identification route
    # than merely named conditional correlations; Qwen still chooses its claim.
    cards=randomized[:12]+observational[:28]
    examples=development_examples(db,randomized[:12])
    examples.extend(development_trajectory_samples(db,observational[:12]))
    result={"protocol":"kuairand27k_open_temporal_explorer_v4",
            "study_goal":"previously unspecified actionable recommendation Bias discovery",
            "dataset":build,"seed":seed,
            "observational_candidates_screened":len(observational),
            "random_assignment_candidates_screened":len(randomized),
            "candidate_concepts":len(cards),
            "public_development_cards":[x["development"] for x in cards[:40]],
            "public_event_examples":examples,
            "scientific_scope":"observational anomaly only; no causal Bias or Debias proved",
            "requirements_to_promote":["independent user/event_day replication",
                                      "controlled intervention or valid identification",
                                      "utility/reference harm and reversal evidence",
                                      "novelty review against existing Bias literature"]}
    (dest/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    # This is evaluator-only material. The Qwen-facing artifact above omits it.
    heldout={"independent_confirmation":[
        {"concept_id":x["development"]["concept_id"],
         "outcome":x["development"]["outcome"],"result":x["confirmation"]}
        for x in cards[:40]]}
    private=dest/"holdout_assessment.json"
    private.write_text(json.dumps(heldout,indent=2)+"\n")
    private.chmod(0o600)
    print(json.dumps({"database":str(db),"events":build["tables"]["events"],
                      "sequence_rows":build["tables"]["anomaly_features"],
                      "candidate_concepts":len(cards),
                      "public_development_top":[{"concept":c["development"]["concept_id"],
                                     "outcome":c["development"]["outcome"],
                                     "dev":c["development"]["observed_difference"]}
                                    for c in cards[:8]]},indent=2))


def qwen_open_discovery(output, model_config):
    """Choose scientific mechanisms using ONLY public development evidence.

    Heldout records remain evaluator-owned and are accessed only after Qwen
    has frozen its candidate IDs and falsification plans.
    """
    from rsi_researcher.research.discovery_cycle import contract,T
    from rsi_researcher.research.scientific_agents import ResearchAgent
    from rsi_researcher.runtime.limits import RunLimits
    from rsi_researcher.runtime.models import create_model

    out=Path(output)
    benchmark=json.loads((out/"results.json").read_text())
    # Prompt budget and arm balance fixed BEFORE the first successful Qwen
    # response. The v1 request exceeded max_model_len; no holdout was read.
    base_cards=benchmark["public_development_cards"]
    randomized=[c for c in base_cards if c.get("source_kind")=="randomized_insertion_assignment"]
    observational=[c for c in base_cards if c.get("source_kind")!="randomized_insertion_assignment"]
    public=randomized[:8]+observational[:6]
    context_samples=benchmark.get("public_event_examples",[])
    if len(public)<2: raise ValueError("Not enough supported development concepts")
    choices=[f"C{i:02d}" for i in range(len(public))]
    model_dir=out/"model_calls"
    model_dir.mkdir(exist_ok=False)
    limits=RunLimits(model_dir,max_requests=5,max_tokens=65000,max_seconds=420)
    model=create_model(model_dir,limits.max_requests,limits,model_config)
    answers=[]
    # One rigorously followed scientific question per experiment.
    for step in range(1):
        allowed=[x for x in choices if x not in [a["candidate_id"] for a in answers]]
        spec=contract({
            "candidate_id":{"type":"string","enum":allowed},
            "novel_explanatory_concept":T,
            "concrete_observation":T,
            "hypothesized_causal_path":T,
            "competing_causes":{"type":"array","items":T,"minItems":2,"maxItems":4},
            "falsification_prediction":T,
            "required_randomized_or_replay_intervention":T,
            "potential_user_or_item_utility_harm":T,
            "reversible_debias_intervention":T,
            "utility_noninferiority_criterion":T,
            "limitation_of_logged_data":T,
            "new_measurement_or_feature_needed":T})
        agent=ResearchAgent(model)
        agent.role="open-sequential-discovery-explorer"
        agent.instruction=(
          "You are an autonomous scientist working on previously UNSPECIFIED "
          "recommender-system Bias mechanisms, not on named classic Biases. "
          "Find a meaningful conditional pattern among real user histories. "
          "Use evidence and rival mechanisms, not plausible-sounding labels. "
          "Some evidence is a randomized VIDEO insertion with downstream "
          "logged outcomes; this does NOT randomize the recommender's internal state "
          "or prove a harmful algorithmic Bias. Verify pretreatment balance and "
          "distinguish platform feedback response from content-driven user effects. "
          "We can inspect prior feedback, novel content, temporal delays and "
          "other users' PRIOR responses. These are observational patterns, "
          "NOT interventions; in particular cross-user correlations do not "
          "establish propagation. Distinguish click prediction from preference "
          "and policy distortion. Explain why a deviation would matter, what "
          "baseline utility is warranted, and a concrete reversible Debias action. "
          "If nothing identifiable is shown, say the claim remains unconfirmed. "
          "Never claim that a new real Bias was verified.")
        candidates=[{"candidate_id":f"C{i:02d}","observation":card,
                     "original_event_examples":[
                         {**x, "raw_event_keys_and_context":x["raw_event_keys_and_context"][:1]}
                         for x in context_samples
                         if x["concept_id"]==card["concept_id"]
                            and x["outcome"]==card["outcome"]][:1]}
                    for i,card in enumerate(public) if f"C{i:02d}" in allowed]
        answer=agent.act({"development_anomalies":candidates,
                          "past_hypotheses":answers,
                          "research_tool_capabilities":[
                             "logged user trajectories","lagged response conditional statistics",
                             "prior-day other-user item feedback",
                             "user-disjoint future-day confirmation",
                             "registered experiments after freezing",
                             "NO arbitrary real-world intervention from offline logs"],
                          "mission":"discover genuinely actionable, previously unspecified Bias mechanisms"},
                          f"kuairand27k-open-{step}",spec)
        answers.append(answer)

    # Freeze model output before accessing confirmation; evaluator checks
    # magnitude, direction, and whether group definitions are supported.
    # Proposals are persisted before any heldout file is loaded.
    (out/"frozen_qwen_proposals.json").write_text(
        json.dumps({"answers":answers},indent=2,ensure_ascii=False)+"\n")
    heldout=json.loads((out/"holdout_assessment.json").read_text())[
        "independent_confirmation"]
    audit=[]
    for answer in answers:
        ix=int(answer["candidate_id"][1:])
        dev=public[ix]
        key=(dev["concept_id"],dev["outcome"])
        confirm=next((x["result"] for x in heldout
                     if (x["concept_id"],x["outcome"])==key),None)
        matched=(bool(confirm)
                 and dev["observed_difference"]*confirm["observed_difference"]>0)
        if dev.get("source_kind")=="randomized_insertion_assignment":
            matched_review={
                "status":"randomized_assignment_ITT_candidate",
                "independent_pretreatment_balance":(confirm or {}).get("pretreatment_balance"),
                "not_a_randomized_algorithm_state_intervention":True,
                "mechanism_identified":False,
                "reason":"the random VIDEO identity is assigned, but internal model state/feedback is not"}
            reference_review={
                "status":"candidate_restricted_random_insertion_assessment",
                "assigned_content_conditional_effect":(confirm or {}).get("observed_difference"),
                "causal_debias_utility_identified":False,
                "note":"Post-assignment feed composition is not proof that ranking utility decreased"}
        else:
            matched_review=matched_independent_review(
                out/"seq.duckdb",dev)
            reference_review=random_reference_audit(out/"seq.duckdb",dev)
        audit.append({
            "qwen_claim":answer,
            "development":dev,
            "independent_user_time_test":confirm,
            "user_and_content_matched_falsification":matched_review,
            "restricted_randomized_feedback_reference":reference_review,
            "same_direction_on_independent_users_and_days":matched,
            "scientific_status":("replicated_observational_concept" if matched
                                 else "insufficient_replication"),
            "mechanism_identified":False,
            "bias_vs_utility_reference_identified":False,
            "debias_effect_verified":False,
            "notes":"A sign agreement on heldout observational logs is not an identified causal mechanism."})
    result={"real_dataset":benchmark["dataset"]["source"],
            "model":"local qwen3-8b",
            "hypotheses":audit,
            "total_model_requests":model.requests,
            "research_conclusion":"open concept generation complete; no causal new Bias accepted without intervention/utility reference"}
    (out/"qwen_hypotheses.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--modulo",type=int,default=32)
    p.add_argument("--seed",type=int,default=20261023)
    p.add_argument("--min-events",type=int,default=180)
    p.add_argument("--qwen-config")
    p.add_argument("--review-existing",action="store_true")
    a=p.parse_args()
    if a.review_existing:
        if not a.qwen_config:
            raise ValueError("--review-existing requires --qwen-config")
        result=qwen_open_discovery(a.output,json.loads(Path(a.qwen_config).read_text()))
        print(json.dumps({"qwen_hypotheses":len(result["hypotheses"]),
                          "model_requests":result["total_model_requests"]}))
    else:
        run(a.dataset,a.output,a.modulo,a.seed,a.min_events)
