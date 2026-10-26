"""Real-data discovery triage on KuaiRand random-exposure logs.

This is an observational concept generator, NOT an identified causal-bias test.
Random-insertion feedback is a *reference for conditional response* on its
supported item pool. No counterfactual recommendation results are invented.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from rsi_researcher.datasets.kuairand import RANDOM_FILE

STANDARD_FILE = "log_standard_4_22_to_5_08_pure.csv"
VIDEO_FILE = "video_features_basic_pure.csv"
USER_FILE = "user_features_pure.csv"

def _rows(file):
    with Path(file).open(newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)

def _split(user_id, date, seed):
    """Different users AND calendar days for evaluation."""
    folded = int.from_bytes(hashlib.blake2b(
        f"{seed}:{user_id}".encode(), digest_size=8).digest(), "big")
    return ("confirmation" if folded % 5 == 0 and date >= 20220501 else
            "development" if folded % 5 != 0 and date <= 20220430 else None)

def _duration(value):
    try: seconds = float(value) / 1000.
    except (TypeError, ValueError): return None
    if not math.isfinite(seconds) or seconds <= 0: return None
    return "short_lt15s" if seconds < 15 else "medium_15to60s" if seconds < 60 else "long_ge60s"

def _data_dir(base):
    base = Path(base)
    for x in (base,base/"data",base/"KuaiRand-Pure"/"data"):
        if (x/RANDOM_FILE).exists(): return x
    raise FileNotFoundError("KuaiRand-Pure official CSV directory not found")

def load_features(data_dir):
    item={}
    for row in _rows(data_dir/VIDEO_FILE):
        tag=row["tag"]
        if not tag or tag.lower()=="nan": tag=None
        item[row["video_id"]]={"tag":tag,"duration":_duration(row.get("video_duration"))}
    users={}
    for row in _rows(data_dir/USER_FILE):
        active=row.get("user_active_degree") or "UNKNOWN"
        users[row["user_id"]]=active
    return item,users

def collect(base, seed, tab="1"):
    folder=_data_dir(base)
    items,users=load_features(folder)
    cells=defaultdict(lambda:[0,0,0,0]) # exposures, click, like, long_view
    totals=defaultdict(lambda:[0,0,0,0])
    counts=defaultdict(int)
    for arm,file in (("random",RANDOM_FILE),("standard",STANDARD_FILE)):
        for row in _rows(folder/file):
            counts["raw_"+arm]+=1
            if row["tab"]!=tab or row["is_rand"]!=("1" if arm=="random" else "0"):
                counts["excluded_tab_or_arm_"+arm]+=1
                continue
            split=_split(row["user_id"],int(row["date"]),seed)
            if split is None:
                counts["excluded_temporal_or_user_"+arm]+=1
                continue
            feature=items.get(row["video_id"])
            if feature is None:
                counts["missing_metadata_"+arm]+=1
                continue
            feedback=[1,int(row["is_click"]),int(row["is_like"]),int(row["long_view"])]
            activity=users.get(row["user_id"],"UNKNOWN")
            # Composable factor × condition cells. No Bias type registry.
            for scope in ("all","activity:"+activity):
                for axis in ("tag","duration"):
                    value=feature[axis]
                    if value is None: continue
                    key=(split,arm,scope,axis)
                    cells[key+(value,)]=[a+b for a,b in zip(cells[key+(value,)],feedback)]
                    totals[key]=[a+b for a,b in zip(totals[key],feedback)]
            counts["eligible_"+split+"_"+arm]+=1
    return dict(cells),dict(totals),dict(counts)

def records_for_split(cells,totals,split,min_random=180,min_standard=45):
    discoveries=[]
    topics=set((k[2],k[3]) for k in totals if k[0]==split)
    for scope,axis in sorted(topics):
        rand=totals.get((split,"random",scope,axis))
        std=totals.get((split,"standard",scope,axis))
        if not rand or not std or rand[0]<400 or std[0]<120: continue
        for key,r in cells.items():
            if key[:4]!=(split,"random",scope,axis):continue
            val=key[4]
            s=cells.get((split,"standard",scope,axis,val),[0,0,0,0])
            n_r,n_s=r[0],s[0]
            if n_r<min_random or n_s<min_standard: continue
            if rand[0]-n_r<180:continue
            ctr=r[1]/n_r
            other=(rand[1]-r[1])/(rand[0]-n_r)
            diff=ctr-other
            se=math.sqrt(max(0,ctr*(1-ctr))/n_r +
                         max(0,other*(1-other))/(rand[0]-n_r))
            share_std=n_s/std[0]
            share_rand=n_r/rand[0]
            merit=(r[1]/rand[1]) if rand[1] else 0
            # Evidence is descriptive, not causal. Give priority to relevant
            # groups that the standard policy underrepresents.
            opportunity=merit-share_std
            predicted_gain=0.02*diff # 2 percentage points shift in group share
            discoveries.append({
              "concept":{"user_condition":scope,"item_attribute":axis,"item_value":val},
              "random_exposures":n_r,"standard_exposures":n_s,
              "random_click_rate":round(ctr,6),
              "other_random_click_rate":round(other,6),
              "random_ctr_advantage":round(diff,6),
              "descriptive_normal95":[round(diff-1.96*se,6),round(diff+1.96*se,6)],
              "random_group_share":round(share_rand,6),
              "standard_group_share":round(share_std,6),
              "click_merit_share":round(merit,6),
              "opportunity_minus_standard_share":round(opportunity,6),
              "illustrative_2pp_reallocation_reward_change":round(predicted_gain,7),
              "supported_nudge":bool(share_std<=0.98 and opportunity>0.02 and diff-1.96*se>0),
              "inference":"observational allocation discrepancy; randomized-insertion conditional CTR reference; not causal policy value"
            })
    return discoveries

def key(row):
    x=row["concept"]
    return (x["user_condition"],x["item_attribute"],x["item_value"])

def run(dataset,out,seed=20261023,limit=12,tab="1"):
    dest=Path(out)
    dest.mkdir(parents=True,exist_ok=False)
    cells,totals,counts=collect(dataset,seed,tab=tab)
    dev=records_for_split(cells,totals,"development")
    hold=records_for_split(cells,totals,"confirmation")
    hold_by_key={key(x):x for x in hold}
    # This is top-k *anomaly triage*, not a claim of novelty or causal impact.
    # No evaluator outcome may influence what is selected.
    plausible=[x for x in dev if x["supported_nudge"]]
    plausible.sort(key=lambda r:(-r["opportunity_minus_standard_share"]*
                               r["random_ctr_advantage"],-r["random_exposures"]))
    hypotheses=[]
    for row in plausible[:limit]:
        other=hold_by_key.get(key(row))
        confirmed=bool(other and other["supported_nudge"] and
                       other["random_ctr_advantage"]>0)
        hypotheses.append({
          "research_claim":{"concept":row["concept"],
             "observation":"random-reference conditional engagement is above other items, yet standard exposure lags within the same scope",
             "mechanism_hypothesis":"the logged allocation may underexpose a higher-engagement content slice",
             "competing_explanations":["item pool and logging policy differ",
                                       "user-by-item relevance heterogeneity",
                                       "time-varying item quality",
                                       "context/position changes click measurement"],
             "proposed_falsification":"match user and available items; randomize group exposure and compare reward under a frozen recommender",
             "debias_handoff":{"candidate_action":"small constrained exposure reallocation, 2pp",
                              "required_reference":"randomized exposure feedback, with adequate support",
                              "utility_guard":"paired heldout reward improvement or noninferiority, with uncertainty",
                              "do_not_run_as_online_policy":True}},
          "development_evidence":row,
          "independent_user_and_time_evidence":other,
          "claim_status":("replicated_descriptive_candidate" if confirmed else
                          "unconfirmed_descriptive_candidate"),
          "mechanism_identified":False,
          "validated_new_bias":False,
          "causal_debias_effect_verified":False
        })
    output={
      "dataset":"KuaiRand-Pure","research_question":"discover actionable previously unspecified exposure/utility anomalies in real logged recommendations",
      "seed":seed,"tab":tab,"data_counts":counts,
      "protocol":{"split":"80% users Apr22-30 for discovery; disjoint 20% users May1-8 for confirmation",
                  "baseline":"randomized inserted exposures of SAME tab, not universal preference truth",
                  "targets":["conditional random CTR advantage","merit share minus standard exposure share"],
                  "composable_concepts":["item.tag × user activity","item duration × user activity","unconditional item slices"],
                  "limitations":["Pure has incomplete standard recommendation sequences",
                                 "nonrandom standard allocation creates selection confounding",
                                 "randomized insertion may have position/context shift",
                                 "normal descriptive intervals do not account for user clusters or sequential searches",
                                 "real exposure difference is not proof of unreasonable Bias",
                                 "reward reallocation is a projected proxy, not an executed policy"]},
      "candidates_screened":len(dev),"passing_development":len(plausible),
      "selected_for_confirmation":len(hypotheses),
      "descriptively_replicated":sum(x["claim_status"]=="replicated_descriptive_candidate" for x in hypotheses),
      "research_claims":hypotheses,
      "scientific_status":"real-data anomaly triage; no discovered causal new bias and no debias effect established"}
    (dest/"results.json").write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"output":str(dest),"data_counts":counts,
        "candidates_screened":len(dev),"dev_positive":len(plausible),
        "selected":len(hypotheses),"replicated":output["descriptively_replicated"],
        "review_samples":[{"concept":x["research_claim"]["concept"],
                           "dev":x["development_evidence"]["opportunity_minus_standard_share"],
                           "heldout": (x["independent_user_and_time_evidence"] or {}).get("opportunity_minus_standard_share"),
                           "status":x["claim_status"]} for x in hypotheses[:6]]},ensure_ascii=False,indent=2))


def heldout_user_cluster_audit(dataset, source_json):
    """Independent heldout user-cluster uncertainty for real-data candidates.

    An event-level interval is not used as confirmation: many impressions
    belong to one user, so uncertainty must account for repeated exposure.
    """
    result=json.loads(Path(source_json).read_text())
    selected=result["research_claims"]
    data_dir=_data_dir(dataset)
    item,users=load_features(data_dir)
    user_cells=[defaultdict(lambda:[0,0,0,0]) for _ in selected]
    for row in _rows(data_dir/RANDOM_FILE):
        if row["tab"]!=result["tab"] or row["is_rand"]!="1":
            continue
        if _split(row["user_id"],int(row["date"]),result["seed"])!="confirmation":
            continue
        features=item.get(row["video_id"])
        if not features: continue
        active=users.get(row["user_id"],"UNKNOWN")
        click=int(row["is_click"])
        for index,candidate in enumerate(selected):
            claim=candidate["research_claim"]["concept"]
            if claim["user_condition"] not in ("all","activity:"+active):
                continue
            is_target=features.get(claim["item_attribute"])==claim["item_value"]
            vals=user_cells[index][row["user_id"]]
            pos=0 if is_target else 2
            vals[pos]+=1
            vals[pos+1]+=click
    summaries=[]
    for index,candidate in enumerate(selected):
        values=list(user_cells[index].values())
        ng=sum(x[0] for x in values)
        yg=sum(x[1] for x in values)
        no=sum(x[2] for x in values)
        yo=sum(x[3] for x in values)
        if not ng or not no: continue
        p=yg/ng;q=yo/no
        effect=p-q
        contributions=[
            (x[1]-p*x[0])/ng-(x[3]-q*x[2])/no
            for x in values]
        m=len(values)
        se=(sum(c*c for c in contributions)*m/(m-1))**.5 if m>1 else float("nan")
        lo=effect-1.96*se
        hi=effect+1.96*se
        independent=candidate.get("independent_user_and_time_evidence")
        replicated=bool(independent and independent.get("supported_nudge") and lo>0)
        summaries.append({
            "concept":candidate["research_claim"]["concept"],
            "n_independent_users":m,
            "n_random_reference_target":ng,
            "n_random_reference_elsewhere":no,
            "heldout_click_advantage":round(effect,6),
            "user_cluster95":[round(lo,6),round(hi,6)],
            "projected_2pp_reallocation_click_change":round(.02*effect,7),
            "projected_gain_cluster95":[round(.02*lo,7),round(.02*hi,7)],
            "independent_descriptive_replication":replicated,
            "causal_debias_verified":False,
            "novel_bias_confirmed":False,
            "interpretation":"heldout user-cluster descriptive interval; no multiple-testing correction; simulated 2pp shift, NOT executed policy"
        })
    return summaries

def qwen_research_review(output,config):
    """Existing Qwen client converts developmental evidence into testable concepts.

    Only development cards reach the model; heldout confirmation stays outside
    model prompts and is attached after model inference.
    """
    from rsi_researcher.runtime.models import create_model
    from rsi_researcher.runtime.limits import RunLimits
    from rsi_researcher.research.scientific_agents import ResearchAgent
    result=json.loads((Path(output)/"results.json").read_text())
    candidates=result["research_claims"]
    public=[{"candidate_id":f"C{i:02d}",
             "condition":x["research_claim"]["concept"],
             "random_reference":x["development_evidence"]}
            for i,x in enumerate(candidates)]
    # Critical: no independent_user_and_time_evidence is passed to the model.
    from rsi_researcher.research.discovery_cycle import contract,T
    spec=contract({"candidate_id":{"type":"string","enum":[c["candidate_id"] for c in public]},
                   "scientific_concept":T,
                   "data_observation":T,
                   "hypothesized_excess_allocation_mechanism":T,
                   "competing_explanations":{"type":"array","items":T,"minItems":2,"maxItems":4},
                   "falsification_experiment":T,
                   "candidate_debias_action":T,
                   "reason_this_matters_for_recommendation_utility":T,
                   "what_current_data_cannot_identify":T})
    model_dir=Path(output)/"model_calls"
    model_dir.mkdir(exist_ok=False)
    limits=RunLimits(model_dir,max_requests=6,max_tokens=30000,max_seconds=350)
    model=create_model(model_dir,limits.max_requests,limits,config)
    ideas=[]
    for rank in range(min(2,len(public))):
        agent=ResearchAgent(model)
        agent.role="real-data-discovery-researcher"
        agent.instruction=(
          "You are discovering potentially NEW useful recommender Bias mechanisms "
          "from REAL KuaiRand randomized-insertion and standard-exposure observations. "
          "Choose one evidence-backed condition combination, give a specific testable "
          "explanation, competing causes, a falsification and a feasible Debias "
          "intervention with a click-utility safeguard. Do not use names of known "
          "Bias categories as discoveries. Numeric video tag IDs are opaque; "
          "DO NOT invent their semantic labels. The shown stats are observational; "
          "no causal policy impact is identified. Pick a different candidate than "
          "earlier rounds; do not claim a confirmed new Bias.")
        previous=[v["candidate_id"] for v in ideas]
        eligible=[v for v in public if v["candidate_id"] not in previous]
        adapted={**spec,"properties":{**spec["properties"],
                   "candidate_id":{"type":"string","enum":[v["candidate_id"] for v in eligible]}}}
        idea=agent.act({"selected_observational_candidates":eligible,
                        "proposed_before":ideas,
                        "referent":"higher randomized-reference click propensity vs lower standard exposure; tab="+result["tab"]},
                       "kuairand-open-concepts-"+str(rank),adapted)
        ideas.append(idea)
    assessed=[]
    for item in ideas:
        i=int(item["candidate_id"][1:])
        candidate=candidates[i]
        assessed.append({
          "qwen_hypothesis":item,
          "developer_evidence":candidate["development_evidence"],
          "heldout_reference":candidate["independent_user_and_time_evidence"],
          "heldout_status":candidate["claim_status"],
          "scientific_status":("replicated descriptive association only" if
                               candidate["claim_status"]=="replicated_descriptive_candidate"
                               else "insufficient independent descriptive support"),
          "causal_debias_effect_verified":False,
          "novel_bias_discovery_confirmed":False
        })
    output_file=Path(output)/"qwen_concepts.json"
    output_file.write_text(json.dumps({
      "model":"existing ResearchAgent via Qwen/vLLM",
      "public_information_access":"development cards only",
      "real_environment":"logged exposures and observed click, not simulated counterfactual ranking",
      "model_requests":model.requests,
      "claims":assessed,
      "note":"Qwen provides candidate concept explanation, not scientific verdict"
    },indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"qwen_result":str(output_file),
                      "selected_candidates":[x["candidate_id"] for x in ideas],
                      "model_calls":model.requests,
                      "heldout_statuses":[x["heldout_status"] for x in assessed]}))
    return assessed

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=20261023)
    p.add_argument("--limit",type=int,default=12)
    p.add_argument("--tab",default="1")
    p.add_argument("--audit-existing",action="store_true")
    p.add_argument("--qwen-config")
    a=p.parse_args()
    if a.audit_existing:
        existing=Path(a.out)/"results.json"
        clusters=heldout_user_cluster_audit(a.dataset,existing)
        (Path(a.out)/"cluster_audit.json").write_text(
            json.dumps({"user_cluster_audit":clusters},indent=2,ensure_ascii=False)+"\n")
        print(json.dumps({"cluster_audit":len(clusters),
            "replicated_with_user_cluster_uncertainty":
            sum(x["independent_descriptive_replication"] for x in clusters)}))
        if a.qwen_config:
            qwen_research_review(a.out,json.loads(Path(a.qwen_config).read_text()))
    else:
        run(a.dataset,a.out,a.seed,a.limit,a.tab)
