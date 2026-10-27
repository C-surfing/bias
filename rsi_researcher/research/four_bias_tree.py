"""Cross-round evidence-guided scientific tree for Selection and Exposure.

Shares the project's real Qwen Diagnosis/Critic/Reflection and runtime.
Numeric evaluator owns all development/confirmation outcomes. The tree owns
frontier, hypothesis lineage, experiments and cross-round evidential memory.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random
from pathlib import Path
from statistics import mean
from rsi_researcher.research.four_bias_benchmarks import (
    SelectionPanel,ExposurePanel,SEEDS,KUAIREC,OBD,BTS,digest)
from rsi_researcher.research.four_bias_contracts import CONTRACTS,assess
from rsi_researcher.research.four_bias_workflow import FrozenBiasCritic
from rsi_researcher.research.scientific_agents import (
    ResearchAgent,DiagnosisAgent,ReflectionAgent,strict,TEXT)
from rsi_researcher.runtime.client import atomic_json,usage_summary
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.models import create_model

ROOT=Path("/data/cjh/rsi/project")
DEFAULTS={
 "selection":{"baseline":{"blend":0.0,"clip":5.0},
              "fixed":{"blend":1.0,"clip":5.0},
              "seed_random":1234},
 "exposure":{"baseline":{"top":20,"mass":.25},
             "fixed":{"top":20,"mass":.9},
             "seed_random":5678}}
# For exposure uniform over all 80 is not representable at top20/mass .25:
# π(head)=.25/20=.0125, π(tail)=.75/60=.0125.
SCHEMA={
 "selection":strict({
    "blend":{"type":"number","minimum":0.0,"maximum":1.0},
    "clip":{"type":"number","minimum":1.0,"maximum":5.0},
    "hypothesis":TEXT,"competing_explanation":TEXT,
    "falsifier":TEXT,"why_parent_failed":TEXT}),
 "exposure":strict({
    "top":{"type":"integer","minimum":5,"maximum":40},
    "mass":{"type":"number","minimum":.25,"maximum":1.0},
    "hypothesis":TEXT,"competing_explanation":TEXT,
    "falsifier":TEXT,"why_parent_failed":TEXT}),
}
class ParametricDesigner(ResearchAgent):
    role="design"
    instruction=(
      "You are the scientific Designer. Propose a NEW typed, executable NUMERIC "
      "policy with legal bounded parameters, not a choice of canned action labels. "
      "Use DEVELOPMENT numeric failure evidence to choose a discriminating change "
      "relative to the selected parent, and state an observable falsifier. "
      "Selection: blend in [0,1] weights observed naive prediction vs corrected "
      "known-MNAR propensity weighted prediction; clip in [1,5] bounds weights. "
      "Exposure: choose 5-40 BTS-frequency ranked items per presentation slot "
      "and give them total probability mass .25-1; the remainder is spread over "
      "other items. The random logger pscore is fixed and cannot be manipulated. "
      "Do NOT infer causal bias removal from a higher SNIPS reward. "
      "Never request test labels, choose hyperparameters based on held-out scores, "
      "or reuse a previously tested parameter fingerprint."
    )

def canon(task,obj):
    if task=="selection":
        val={"blend":round(float(obj["blend"]),3),
             "clip":round(float(obj["clip"]),3)}
        if not 0<=val["blend"]<=1 or not 1<=val["clip"]<=5:raise ValueError(val)
    elif task=="exposure":
        val={"top":int(obj["top"]), "mass":round(float(obj["mass"]),3)}
        if not 5<=val["top"]<=40 or not .25<=val["mass"]<=1:raise ValueError(val)
    else:raise ValueError(task)
    return val

def identity(params):
    return json.dumps(params,sort_keys=True,separators=(",",":"))

def panel_for(task,phase):
    return SelectionPanel(phase) if task=="selection" else ExposurePanel(phase)

def score_dev(task,panel,baseline,params,seeds=SEEDS):
    comparisons=[panel.compare(baseline,params,seed) for seed in seeds]
    checks=[]
    for c in comparisons:
        result=assess(task,c["base"],c["candidate"],
                    identification=panel.identification,evidence_grade="development_only",
                    ess=c["candidate"].get("ESS"))
        uncertainty=c["CI95_gain"]
        if task=="selection":
            utility=(c["candidate"]["NDCG@50"]>=.95*c["base"]["NDCG@50"])
            supported=(c["user_blocks"]>=100)
        else:
            utility=c["candidate"]["ESS"]>=c["min_ESS_required"]
            supported=(c["time_blocks"]>=50 and c["candidate"]["clicks"]>=200)
        checks.append({"registered":result,"support":supported,
          "CI95_gain":uncertainty,"confidence_direction_positive":
          uncertainty is not None and uncertainty[0]>0,
          "utility_or_ESS":bool(utility)})
    mean_gain=mean(c["delta_primary"] for c in comparisons)
    all_supported=all(c["support"] and c["registered"]["identification_ok"] for c in checks)
    all_quality=all(c["utility_or_ESS"] for c in checks)
    all_directions=all(c["confidence_direction_positive"] for c in checks)
    mean_target=mean(c["candidate"][CONTRACTS[task].metric] for c in comparisons)
    # Development screening is explicitly conservative; no sequential FWER control.
    eligible=(all_supported and all_quality and all_directions
              and mean_gain>CONTRACTS[task].min_gain)
    if not all_supported:mode="insufficient_identification_or_support"
    elif not all_quality:mode="utility_or_ESS_violation"
    elif not all_directions:mode="unstable_or_unresolved_effect"
    elif not eligible:mode="target_not_improved_enough"
    else:mode="development_candidate_only"
    return {
       "target":task,"parametric_strategy":params,"seeds":list(seeds),
       "mean_baseline":mean(c["base"][CONTRACTS[task].metric] for c in comparisons),
       "mean_candidate":mean_target,"mean_primary_gain":mean_gain,
       "per_seed":[{"seed":seed,"base":c["base"],"candidate":c["candidate"],
         "delta_primary":c["delta_primary"],"CI95_gain":c["CI95_gain"],
         "utility_or_ESS":flag["utility_or_ESS"],"support":flag["support"]}
         for seed,c,flag in zip(seeds,comparisons,checks)],
       "development_eligible":bool(eligible),"failure_mode":mode,
       "independent_confirmation":False,
       "causal_bias_reduction_proven":False,
       "scientific_scope":("only synthetic extra selection MNAR corrected" if task=="selection"
                           else "static policy reward under logged random action propensity")
    }

def random_strategy(task,rng):
    if task=="selection":
        return canon(task,{"blend":rng.uniform(.15,1),"clip":rng.uniform(1.2,5)})
    return canon(task,{"top":rng.randint(5,40),"mass":rng.uniform(.35,1)})

def choose_parent(nodes):
    # Measured node first when it reveals a falsifiable failure; do not
    # blindly chase a scalar composite. Preserve ROOT as exploration fallback.
    real=[n for n in nodes if n["id"]!="ROOT" and n["status"]=="MEASURED"]
    possible=[n for n in real if n["depth"]<3 and n["expansions"]<2]
    if not possible:return nodes[0]
    uncertain={"unstable_or_unresolved_effect":3,
               "utility_or_ESS_violation":2,
               "target_not_improved_enough":1,
               "development_candidate_only":0}
    chosen=max(possible,key=lambda n:(
        uncertain.get(n["audit"]["failure_mode"],0),
        -n["expansions"],-n["depth"],n["audit"]["mean_primary_gain"]))
    # Explore root every third turn rather than collapse to an unvalidated leaf.
    return nodes[0] if len(real)%3==2 else chosen

def freeze_metadata(task,output,prior):
    source_files=([KUAIREC] if task=="selection" else [OBD,BTS])
    return {"protocol":"four_bias_scientific_tree_frozen_v2",
       "task":task,"source_hashes":{p.name:digest(p) for p in source_files},
       "cohort_split":"KuaiRec users stable 70/20/10 train/dev/confirm"
          if task=="selection" else "OBD full random/all disjoint row-index 80/20 dev/confirmation",
       "development_seeds":list(SEEDS),"holdout_access_before_freeze":False,
       "search_policy":"failure_guided_expansion_with_root_exploration",
       "editable_policy":"numeric selection blend/clip or OBD top/mass",
       "immutable":["source_data","splits","label_reference","estimator","utility_guard"],
       "prior_campaign":str(prior) if prior else None,
       "causal_claim":False}

def research(task,output,config,*,nodes_limit=4,prior_campaign=None,
             confirm=True,random_trials=4):
    if task not in ("selection","exposure"):raise ValueError(task)
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    if not 1<=nodes_limit<=6:raise ValueError("bounded nodes 1..6")
    baseline=DEFAULTS[task]["baseline"];fixed=DEFAULTS[task]["fixed"]
    meta=freeze_metadata(task,out,prior_campaign)
    prior_nodes=[]
    prior_random=[]
    if prior_campaign:
        prior=Path(prior_campaign).resolve()
        manifest=json.loads((prior/"protocol.json").read_text())
        if manifest["protocol"]!=meta["protocol"] or manifest["task"]!=task or (
            manifest["source_hashes"]!=meta["source_hashes"] or
            manifest["development_seeds"]!=meta["development_seeds"]):
            raise ValueError("scientific prior campaign frozen protocol mismatch")
        old=json.loads((prior/"summary.json").read_text())
        if old.get("holdout_opened") or (prior/"confirmation.json").exists():
            raise ValueError("never adaptively reuse opened confirmation evidence")
        prior_nodes=[n for n in old["tree"] if n["id"]!="ROOT"]
        prior_random=old["random_search"]
    out.mkdir(parents=True)
    (out/"nodes").mkdir()
    atomic_json(out/"protocol.json",meta)
    panel=panel_for(task,"dev")
    print("FROZEN_DEV_LOADED",task,{"events":(panel.n if task=="exposure"
           else len(panel.v)),"users":len(panel.users) if task=="selection" else None,
           "train_events":len(panel.item) if task=="selection" else None},flush=True)
    base_per_seed=[panel.measure(baseline,s) for s in SEEDS]
    fixed_audit=score_dev(task,panel,baseline,fixed)
    # Advance the random baseline across rounds, and keep every prior trial.
    rng=random.Random(DEFAULTS[task]["seed_random"]+20261022)
    for _ in range(len(prior_random)):
        random_strategy(task,rng)
    random_arms=[{"id":r["id"],"strategy":canon(task,r["strategy"]),
                  "audit":score_dev(task,panel,baseline,canon(task,r["strategy"])),
                  "inherited_from":str(prior_campaign),
                  "scientist_model_calls":0} for r in prior_random]
    seen={identity(baseline),identity(fixed)}
    seen.update(identity(x["strategy"]) for x in random_arms)
    for trial in range(len(random_arms),len(random_arms)+random_trials):
        for _ in range(50):
            params=random_strategy(task,rng)
            if identity(params) not in seen:break
        seen.add(identity(params))
        random_arms.append({"id":f"R{trial+1:03d}","strategy":params,
              "audit":score_dev(task,panel,baseline,params),
              "scientist_model_calls":0})
    nodes=[{"id":"ROOT","parent":None,"depth":0,
            "strategy":baseline,"status":"BASELINE","expansions":0,
            "audit":{"failure_mode":"not_evaluated","per_seed":base_per_seed}}]
    for old in prior_nodes:
        params=canon(task,old["strategy"])
        if identity(params) in seen:continue
        # Re-evaluate archived development on EXACT frozen dev and data hashes,
        # never treating inherited numeric claims as independent confirmation.
        frozen=score_dev(task,panel,baseline,params)
        nodes.append({"id":f"P{len(nodes):03d}","parent":"ROOT",
            "depth":1,"strategy":params,"status":"MEASURED",
            "expansions":0,"audit":frozen,"inherited_from":str(prior_campaign)})
        seen.add(identity(params))
    limit=RunLimits(out,max_requests=min(100,config["budget"]["max_requests"]),
                    max_tokens=min(400000,config["budget"]["max_tokens"]),
                    max_seconds=min(4800,config["budget"]["max_seconds"]))
    model=create_model(out,limit.max_requests,limit,config)
    model.health()
    atomic_json(out/"fixed_and_random.json",{
        "baseline":baseline,"baseline_per_seed":base_per_seed,
        "fixed":{"strategy":fixed,"audit":fixed_audit},
        "random":random_arms,
        "comparison_budget":{"random_numeric_evaluations":random_trials,
          "agent_numeric_evaluations":nodes_limit,
          "strict_equal_Qwen_compute":False},
        "holdout":None})
    history=[]
    for i in range(nodes_limit):
        parent=choose_parent(nodes)
        parent["expansions"]+=1
        # Scientist sees DEVELOPMENT aggregates only, not row-level feedback
        # or the target group's hidden confirmation data.
        history_brief=[{"id":n["id"],"parent":n["parent"],
           "strategy":n["strategy"],"mean_primary_gain":
               n["audit"].get("mean_primary_gain"),"failure_mode":
               n["audit"].get("failure_mode"),"development_eligible":
               n["audit"].get("development_eligible",False)}
               for n in nodes[-6:]]
        obs={"bias_target":task,"registered_contract":CONTRACTS[task].__dict__,
          "research_phase":"development","parent_id":parent["id"],
          "parent_strategy":parent["strategy"],
          "parent_development_audit":parent["audit"],
          "baseline_summary_by_seed":base_per_seed,
          "frozen_reference":meta["cohort_split"],"past_numeric_experiments":history_brief,
          "missing_evidence":"independent confirmation not opened; no row-level outcomes",
          "no_test_labels":True,
          "critic_standard":"the development experiment may proceed without holdout"}
        diag=DiagnosisAgent(model).act({"observations":obs},f"{task}-N{i+1}-diag")
        proposals=[]
        accepted=None
        for revision in range(2):
            plan=ParametricDesigner(model).act({
                "observations":obs,"diagnosis":diag,
                "prior_critic_feedback":proposals[-1].get("critic") if proposals else None,
                "existing_fingerprints":list(seen)[-24:]},
                f"{task}-N{i+1}-design-r{revision}",schema=SCHEMA[task])
            params=canon(task,plan)
            structural=identity(params) in seen
            critic=FrozenBiasCritic(model).act({
                 "candidate":plan,"diagnosis":diag,"parent_strategy":parent["strategy"],
                 "structural_error":"exact_duplicate" if structural else None,
                 "known_data_identification":panel.identification,
                 "task_scope":CONTRACTS[task].identification,
                 "phase":"development only; causal claims forbidden"},
                 f"{task}-N{i+1}-crit-r{revision}")
            proposals.append({"proposal":plan,"critic":critic,
                               "structural_duplicate":structural})
            if critic["decision"]=="approve" and not structural:
                accepted=params
                break
        node_id=f"N{i+1:03d}"
        if accepted:
            audit=score_dev(task,panel,baseline,accepted)
            node={"id":node_id,"parent":parent["id"],"depth":parent["depth"]+1,
                  "strategy":accepted,"status":"MEASURED","expansions":0,
                  "audit":audit,"requests":model.requests}
            seen.add(identity(accepted))
        else:
            audit={"failure_mode":"critic_or_duplicate_rejected",
                   "development_eligible":False}
            node={"id":node_id,"parent":parent["id"],"depth":parent["depth"]+1,
                  "strategy":params,"status":"REJECTED","expansions":0,
                  "audit":audit,"requests":model.requests}
        reflection=ReflectionAgent(model).act({
            "independent_numeric_development_audit":audit,
            "diagnosis":diag,"proposal_revisions":proposals,
            "prev_results":history_brief,
            "evidence_authority":"python numeric only, not agent",
            "scientific_claim":"never independently confirmed",
            "cost_requests_before_reflection":model.requests},f"{task}-N{i+1}-reflect")
        nodes.append(node)
        event={"node":node_id,"parent":parent["id"],"diagnosis":diag,
             "proposals":proposals,"audit":audit,"reflection":reflection,
             "candidate_research_skill":"tentative_not_independently_verified"}
        history.append(event)
        atomic_json(out/"nodes"/(node_id+".json"),event)
        atomic_json(out/"summary.json",{"status":"development_running",
           "task":task,"nodes":nodes,"selection_history":history,
           "qwen_model_requests":model.requests,"holdout_opened":False,
           "confirmed_discoveries":0})
        print("RSI_TREE_NODE",task,node_id,"parent",parent["id"],
              "strategy",accepted,"status",audit["failure_mode"],
              "requests",model.requests,flush=True)
    # Candidate must satisfy ALL registered multi-seed development gates,
    # plus per-seed uncertainty lower bound, before selection is frozen.
    eligible=[{"arm":"fixed","strategy":fixed,"audit":fixed_audit}]
    eligible+= [{"arm":n["id"],"strategy":n["strategy"],"audit":n["audit"]}
                for n in nodes if n["id"]!="ROOT" and n["status"]=="MEASURED"]
    eligible+= [{"arm":n["id"],"strategy":n["strategy"],"audit":n["audit"]}
                for n in random_arms]
    qualified=[x for x in eligible if x["audit"]["development_eligible"]]
    winner=max(qualified,key=lambda x:x["audit"]["mean_primary_gain"]) if qualified else None
    # Freeze all method-specific winners before opening a single confirmation.
    def arm_choice(label,items):
        possible=[x for x in items if x["audit"]["development_eligible"]]
        return max(possible,key=lambda x:x["audit"]["mean_primary_gain"]) if possible else None
    chosen={"fixed":arm_choice("fixed",[eligible[0]])}
    # Candidate groups are selected explicitly, never by a fragile node slice.
    chosen["random"]=arm_choice("random",[
        {"arm":n["id"],"strategy":n["strategy"],"audit":n["audit"]} for n in random_arms])
    chosen["agent"]=arm_choice("agent",[
        {"arm":n["id"],"strategy":n["strategy"],"audit":n["audit"]}
        for n in nodes if n["id"]!="ROOT" and n["status"]=="MEASURED"])
    seal={"task":task,"phase":"frozen_before_confirmation",
        "candidate":winner,"method_winners":chosen,
        "confirmation_allowed":bool(confirm and winner),
        "searched_nodes":nodes_limit,"development_seeds":list(SEEDS),
        "guard":"If no eligible strategy, never read confirmation labels",
        "old_dataset_historical_exploration_may_overlap_confirmation_users":
             task=="selection",
        "scientific_claim_scope":("extra synthetic MNAR correction only"
            if task=="selection" else "OPE static policy click value, not exposure debias reduction"),
        "freeze_dataset_hashes":meta["source_hashes"],
        "qwen_requests_before_holdout":model.requests}
    atomic_json(out/"frozen_before_confirmation.json",seal)
    confirmation=None
    if seal["confirmation_allowed"]:
        unseen=panel_for(task,"confirm") # only now read independent labels
        arms={"baseline":baseline}
        for k,v in chosen.items():
            if v:arms[k]=v["strategy"]
        confirmations={}
        for label,params in arms.items():
            if label=="baseline":
                confirmations[label]={"strategy":params,
                   "by_seed":[unseen.measure(params,seed) for seed in SEEDS]}
            else:
                cs=[unseen.compare(baseline,params,seed) for seed in SEEDS]
                confirmations[label]={"strategy":params,"by_seed":cs,
                 "consistent_positive_CI":all(c["CI95_gain"][0]>0 for c in cs),
                 "utility_guard_pass":all(
                    c["candidate"]["NDCG@50"]>=.95*c["base"]["NDCG@50"]
                    if task=="selection" else
                    c["candidate"]["ESS"]>=c["min_ESS_required"] for c in cs)}
        confirmation={"phase":"opened_once_after_freeze","task":task,
           "cohort":"disjoint user reference" if task=="selection"
                 else "disjoint full-OBD randomized impressions",
           "summary":confirmations,"confirmed_bias_causal_claims":0,
           "independent_confirmation_of_specific_estimand":{
             label:bool(row.get("consistent_positive_CI") and row.get("utility_guard_pass"))
             for label,row in confirmations.items() if label!="baseline"},
           "scope":"known synthetic MNAR effect only" if task=="selection"
                 else "static randomized-logger OPE value, NOT bias reduction",
           "not_global_blindness":task=="selection"}
        atomic_json(out/"confirmation.json",confirmation)
    final={"status":"completed","protocol":meta,"task":task,
      "development_seeds":list(SEEDS),"baseline":base_per_seed,
      "fixed":{"strategy":fixed,"audit":fixed_audit},
      "random_search":random_arms,"tree":nodes,"selection_history":history,
      "pre_confirmation_freeze":seal,
      "confirmation":confirmation,
      "holdout_opened":confirmation is not None,
      "qwen_requests":model.requests,
      "numeric_evaluations":{"fixed":1,"random":len(random_arms),
                             "agent":nodes_limit},
      "strict_equal_compute":False,
      "confirmed_general_debias_discoveries":0,
      "cost_per_confirmed_general_discovery":None}
    atomic_json(out/"summary.json",final)
    atomic_json(out/"usage.json",usage_summary(out))
    print("FOUR_BIAS_TREE_FINAL",json.dumps({
        "task":task,"qwen_calls":model.requests,
        "candidate":winner["arm"] if winner else None,
        "confirmed_estimand":(confirmation or {}).get(
            "independent_confirmation_of_specific_estimand"),
        "holdout_opened":final["holdout_opened"],
        "general_debias_discoveries":0},ensure_ascii=False),flush=True)
    return final

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--task",required=True,choices=("selection","exposure"))
    p.add_argument("--output",required=True)
    p.add_argument("--nodes",type=int,default=4)
    p.add_argument("--random-trials",type=int,default=4)
    p.add_argument("--prior")
    p.add_argument("--development-only",action="store_true")
    args=p.parse_args(argv)
    config=json.loads((ROOT/"configs/llmrank_qwen.json").read_text())
    research(args.task,args.output,config,nodes_limit=args.nodes,
             random_trials=args.random_trials,prior_campaign=args.prior,
             confirm=not args.development_only)
if __name__=="__main__":main()
