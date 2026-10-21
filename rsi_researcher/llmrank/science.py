"""Mainline RSI scientific tree adapter: LLMRank & Amazon intervention research.

Runs under the EXISTING science-tree CLI, VLLMModel, RunLimits, and the
Diagnosis/Designer/Critic/Reflection agent contracts. Dataset and debias
strategies are task adapters, not a parallel research product.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import time
from rsi_researcher.llmrank.agents import (
    RankGateway, LLMRankDesignerAgent, LLMRankCriticAgent,
    DiagnosisAgent, ReflectionAgent
)
from rsi_researcher.llmrank.dataset import load_dataset, cases_for
from rsi_researcher.llmrank.tools import candidate_snapshot,metric_tradeoff
from rsi_researcher.research.scientific_skills import proposal as record_skill
from rsi_researcher.llmrank.candidates import (
    cases_from_source, source_candidates
)
from rsi_researcher.llmrank.strategy import (
    BASELINE, FIXED, canonical, fingerprint, evaluate_strategy, bootstrap_paired
)
from rsi_researcher.research.scientific_agents import authoritative_evidence
from rsi_researcher.runtime.client import atomic_json, usage_summary
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.models import create_model, model_description

def utc():
    return datetime.now(timezone.utc).isoformat()

def select_case_panels(dataset, data, retrieval, n, seed, fraction=0.6):
    train,positive,known,titles,pop=load_dataset(data,dataset)
    source=Path(data)/f"{dataset}.{retrieval}"
    if dataset in ("Beauty","Sports","CDs"):
        source=None
        users=sorted(train)
    else:
        if not source.is_file():
            raise FileNotFoundError(f"author candidate file missing: {source}")
        source_users=source_candidates(source)
        users=[u for u in sorted(train) if u in source_users]
    if len(users)<n:
        raise ValueError(f"need {n} eligible users, found {len(users)}")
    random.Random(seed).shuffle(users)
    requested=users[:n]
    split=int(n*fraction)
    if source:
        dev,skip_dev=cases_from_source(dataset,"validation",requested[:split],
              train,positive,titles,source,seed,20,50)
        test,skip_test=cases_from_source(dataset,"test",requested[split:],
              train,positive,titles,source,seed+1,20,50)
    else:
        dev=cases_for("validation",requested[:split],train,positive,known,titles,seed,20,retrieval)
        test=cases_for("test",requested[split:],train,positive,known,titles,seed+1,20,retrieval)
        skip_dev=skip_test={}
    if len(dev)<2 or len(test)<2:
        raise ValueError("insufficient nonempty development and test users")
    with (Path(data)/f"{dataset}.inter.zip").open("rb") as f:
        data_sha256=hashlib.file_digest(f,"sha256").hexdigest()
    protocol={"dataset":dataset,"seed":seed,"eligible_users":len(users),
          "dev_users":[c["user"] for c in dev],
          "holdout_users":[c["user"] for c in test],"dev_count":len(dev),
          "holdout_count":len(test),"candidate_source":source.name if source else "frozen_random",
          "retrieval":retrieval,
          "candidate_generation":("author_precomputed_"+retrieval if source
                 else "train_title_bm25" if retrieval=="bm25" else "frozen_random"),
          "budget_candidates":20,
          "official_LLMRank_full_reproduction":False,
          "candidate_position_protocol":"target-appended, frozen shuffled candidates, paired gt-first presentation diagnosis",
          "history_len":50,
          "skipped":{"dev":skip_dev,"test":skip_test},
          "data_archive_sha256":data_sha256}
    return dev,test,titles,pop,protocol

def audit(baseline,candidate,*, relative_utility_tolerance=0.05):
    b=baseline["5"];c=candidate["5"]
    utility=(c["NDCG@5"] >= (1-relative_utility_tolerance)*b["NDCG@5"]
         and c["HR@5"] >= b["HR@5"]-max(0.03,1/max(1,b["users"])))
    validity=(c["invalid_rankings"]<=b["invalid_rankings"] and c["valid_rankings"]>0)
    changes={
      "NDCG@5":c["NDCG@5"]-b["NDCG@5"],
      "HR@5":c["HR@5"]-b["HR@5"],
      "ARP@5":(c["ARP@5"]-b["ARP@5"] if c["ARP@5"] is not None and b["ARP@5"] is not None else None),
      "APLT@5":(c["APLT@5"]-b["APLT@5"] if c["APLT@5"] is not None and b["APLT@5"] is not None else None),
    }
    pos_b=b.get("CandidatePositionBias_NDCG@5")
    pos_c=c.get("CandidatePositionBias_NDCG@5")
    improvements={
      "lower_arp":changes["ARP@5"] is not None and changes["ARP@5"] < -0.02*max(1,b["ARP@5"]),
      "higher_aplt":changes["APLT@5"] is not None and changes["APLT@5"]>0.02,
      "less_candidate_position_bias": pos_b is not None and pos_c is not None and abs(pos_c)<abs(pos_b)-0.02,
    }
    eligible=utility and validity and any(improvements.values())
    # Development-only merit: a better trade-off without collapsing utility.
    merit=(
       0.25*(max(0,-changes["ARP@5"])/max(1,b["ARP@5"]) if changes["ARP@5"] is not None else 0)
       +0.5*(max(0,changes["APLT@5"]) if changes["APLT@5"] is not None else 0)
       +(max(0,abs(pos_b)-abs(pos_c)) if pos_b is not None and pos_c is not None else 0)*0.15
       +changes["NDCG@5"]*0.2
    ) if eligible else -1
    return {"eligible_development_only":eligible,"utility_guard":utility,
            "validity_guard":validity,"metric_deltas":changes,
            "improvements":improvements,"merit":merit,
            "not_causal_or_confirmatory":True}

def expansion_priority(baseline,current):
    """Preserve promising bias mechanisms even if utility gate fails."""
    if not current:
        return -10.0
    b,c=baseline["5"],current["5"]
    if None in (b["ARP@5"],c["ARP@5"],b["APLT@5"],c["APLT@5"]):
        return -1.0
    pop_gain=(b["ARP@5"]-c["ARP@5"])/max(b["ARP@5"],1)
    tail_gain=c["APLT@5"]-b["APLT@5"]
    quality_loss=max(0,b["NDCG@5"]-c["NDCG@5"])/max(b["NDCG@5"],0.1)
    return 0.4*pop_gain+0.5*tail_gain-0.6*quality_loss

def search(output,*,config,data,dataset,retrieval,users,seeds,max_nodes=2,
           max_requests=800,arm="qwen_audited_history",prior_campaign=None,
           development_only=False):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if dataset not in ("ml-1m","Games","Beauty","Sports","CDs"):
        raise ValueError("unregistered dataset")
    if not 6<=users<=6000 or not 1<=max_nodes<=6 or max_requests<10:
        raise ValueError("invalid bounded protocol")
    data=Path(data).resolve()
    prior_development=[]
    inherited_by_seed={}
    prior_rank_cache_paths=[]
    visited=set()
    prior_path=Path(prior_campaign).resolve() if prior_campaign else None
    while prior_path:
        if not prior_path.is_dir() or prior_path==output or prior_path in visited:
            raise ValueError("invalid or cyclic prior campaign lineage")
        visited.add(prior_path)
        manifest=prior_path/"program.json"
        prior_rank_cache_paths.append(prior_path/"rank_cache.jsonl")
        if not manifest.is_file():
            raise ValueError("missing original run program manifest")
        program=json.loads(manifest.read_text(encoding="utf-8"))
        if program.get("development_only") is not True:
            raise ValueError("cannot reuse development evidence from a holdout-opened campaign")
        for source in sorted(prior_path.glob("seed-*/result.json")):
            archive=json.loads(source.read_text(encoding="utf-8"))
            if archive["cohort"]["dataset"]!=dataset or archive["holdout"] is not None:
                raise ValueError("prior campaign contains incompatible or holdout evidence")
            if archive["cohort"]["seed"] not in inherited_by_seed:
                inherited_by_seed[archive["cohort"]["seed"]]={
                    "cohort":archive["cohort"],
                    "selected":archive["development"]["selected_rsi"],
                }
            for n in archive["development"]["tree"]:
                if n.get("status")=="MEASURED":
                    prior_development.append({"strategy":n["strategy"],
                      "audit":n["audit"],"origin":str(source),
                      "grade":"development_only_not_independent"})
        parent=program.get("prior_campaign")
        prior_path=Path(parent).resolve() if parent else None
    output.mkdir(parents=True)
    run_budget={**config["budget"],"max_requests":min(config["budget"]["max_requests"],max_requests)}
    limits=RunLimits(output,**run_budget)
    model=create_model(output,limits.max_requests,limits,config)
    model.health()
    gateway=RankGateway(model,output/"rank_cache.jsonl",
                       preload_paths=reversed(prior_rank_cache_paths))
    atomic_json(output/"program.json",{
       "goal":"Maximize independently measured debias quality with controlled utility loss",
       "editable":"strategy instruction, history framing, candidate presentation, votes, rerank formula",
       "immutable":"evaluation, candidates, labels, split, metrics, model, time and request budget",
       "source_branch":"mainline RSI science-tree","author_harness_reused":True,
       "agent_roles":["DiagnosisAgent","LLMRankDesignerAgent",
                      "LLMRankCriticAgent","ReflectionAgent"],
       "model":model_description(config),"seeds":seeds,"users":users,
       "node_limit":max_nodes,"budget":run_budget,"arm":arm,
       "no_test_labels_in_agent_context":True,
       "prior_development_records":len(prior_development),
       "prior_frozen_rank_cache_files":[str(x) for x in prior_rank_cache_paths],
       "prior_campaign":str(Path(prior_campaign).resolve()) if prior_campaign else None,
       "development_only":development_only})
    summaries=[]
    for seed in seeds:
        rd=output/f"seed-{seed}"
        rd.mkdir()
        dev,test,titles,pop,protocol=select_case_panels(dataset,data,retrieval,users,seed)
        if seed in inherited_by_seed:
            prev=inherited_by_seed[seed]["cohort"]
            for field in ("seed","dataset","dev_users","holdout_users",
                          "data_archive_sha256","retrieval","candidate_source"):
                if prev.get(field)!=protocol.get(field):
                    raise ValueError("cross-round frozen cohort mismatch: "+field)
        atomic_json(rd/"cohort.json",protocol)
        # Stored ground-truth labels are evaluator-only; research agents receive
        # aggregate metrics and leakage-safe tool snapshots, NEVER this file.
        atomic_json(rd/"frozen_evaluator_only.json",{"dev":dev,"test":test})
        tool_evidence=candidate_snapshot(dev,pop)
        atomic_json(rd/"diagnostic_tools.json",tool_evidence)
        eval_cache={}
        def run_exp(name,strategy,cases,prefix):
            key=(prefix,fingerprint(strategy))
            if key in eval_cache:
                cached=eval_cache[key]
                atomic_json(rd/f"{prefix}_{name}.json",{
                    "strategy":strategy,"metrics":cached["metrics"],
                    "requests":0,"reused_frozen_identical_inference":True})
                return cached,0
            start=gateway.calls
            result=evaluate_strategy(gateway,cases,titles,pop,strategy,seed,name)
            eval_cache[key]=result
            atomic_json(rd/f"{prefix}_{name}.json",{
                 "strategy":strategy,"metrics":result["metrics"],
                 "rows_random":result["random"],"rows_first":result["first"],
                 "requests":gateway.calls-start})
            return result,gateway.calls-start
        baseline,cost_base=run_exp("plain",BASELINE,dev,"dev")
        fixed_start=gateway.calls
        fixed=[]
        for name,strategy in list(FIXED.items())[:max_nodes]:
            measured,used=run_exp(name,canonical(strategy),dev,"dev")
            outcome=audit(baseline["metrics"],measured["metrics"])
            fixed.append({"name":name,"strategy":canonical(strategy),
                         "audit":outcome,"metrics":measured["metrics"],
                         "requests":used})
        fixed_end=gateway.calls
        best_fixed=max([{"name":"plain","strategy":BASELINE,
                         "audit":{"eligible_development_only":False,"merit":0},
                         "metrics":baseline["metrics"],"requests":0},*fixed],
                         key=lambda x:x["audit"]["merit"])
        # Mainline tree owns hypothesis lineage and shared scientist roles.
        nodes=[{"id":"ROOT","parent":None,"depth":0,"strategy":BASELINE,
                "status":"BASELINE","merit":0,"search_priority":0.0,
                "expansions":0,"metrics":baseline["metrics"]}]
        best=nodes[0]
        seen={fingerprint(BASELINE)}
        seen.update(fingerprint(row["strategy"]) for row in fixed)
        seen.update(fingerprint(row["strategy"]) for row in prior_development)
        inherited=inherited_by_seed.get(seed)
        if inherited:
            strategy=canonical(inherited["selected"]["strategy"])
            if fingerprint(strategy)!=fingerprint(BASELINE):
                measured,cost=run_exp("inherited",strategy,dev,"dev")
                assessment=audit(baseline["metrics"],measured["metrics"])
                inherited_node={
                    "id":"INHERITED","parent":"ROOT","depth":1,
                    "strategy":strategy,"status":"MEASURED",
                    "merit":assessment["merit"],
                    "search_priority":expansion_priority(baseline["metrics"],measured["metrics"]),
                    "expansions":0,
                    "metrics":measured["metrics"],"audit":assessment,
                    "requests":cost,
                    "parent_campaign":str(prior_campaign),
                }
                nodes.append(inherited_node)
                if assessment["eligible_development_only"] and assessment["merit"]>0:
                    best=inherited_node
        history=[]
        tree_start=gateway.calls
        for turn in range(1,max_nodes+1):
            parent=max((n for n in nodes if n.get("status") in ("BASELINE","MEASURED")
                        and n["depth"]<max_nodes),
                       key=lambda n:(n.get("search_priority",0)-0.015*n.get("expansions",0),
                                     -n["depth"]))
            parent["expansions"]=parent.get("expansions",0)+1
            experience=[{"id":node["id"],"parent":node["parent"],
                 "metrics":node["metrics"],"strategy":node["strategy"],
                 "status":node["status"]} for node in nodes]
            observation={"task":"llmrank_bias","phase":"development",
                "registered_metrics":["NDCG@5","HR@5","ARP@5","APLT@5",
                       "CandidatePositionBias_NDCG@5","PopREO@5"],
                "baseline_metrics":baseline["metrics"],
                "candidate_distribution_tool_result":tool_evidence,
                "prior_verified_numeric_development_memory":prior_development[-4:],
                "previous_measured_development_experiments":experience[-5:],
                "recent_critic_and_reflection_feedback":history[-2:],
                "previously_tested_strategy_fingerprints":sorted(seen),
                "candidate_parent_strategy":parent["strategy"],
                "search_guidance":"Avoid exact repeats. When a parent lowers ARP "
                    "but damages NDCG, try reducing strength, mixing with base "
                    "or changing fewer dimensions to preserve utility.",
                "parent_id":parent["id"],
                "scientific_question":"Find a testable LLM ranking intervention while preserving recommendation utility",
                "utility_floor":"development NDCG no more than 5% relative loss; HR at most one sampled hit loss",
                "no_holdout_information":True}
            diag=DiagnosisAgent(model).act({"observations":observation},f"seed{seed}-node{turn}")
            proposals=[]
            for revision in range(2):
                plan=LLMRankDesignerAgent(model).propose(
                    {"diagnosis":diag,"development_evidence":observation,
                     "previous_revision":proposals[-1] if proposals else None},f"seed{seed}-node{turn}-rev{revision}")
                try:
                    strategy=canonical(plan["strategy"])
                    from rsi_researcher.llmrank.formula import validate as validate_formula
                    validate_formula(strategy["score_expression"])
                    structural_error=None
                    if fingerprint(strategy) in seen:
                        raise ValueError("duplicate previously evaluated strategy; prune exact repeat")
                except (ValueError,TypeError,KeyError) as exc:
                    strategy=None
                    structural_error=str(exc)
                review=LLMRankCriticAgent(model).act({
                      "proposal":plan,"diagnosis":diag,"structural_error":structural_error,
                      "paired_protocol":protocol,"condition":"development-only, no causal claim"},f"seed{seed}-node{turn}-rev{revision}")
                proposals.append({"plan":plan,"review":review,"error":structural_error})
                if strategy is not None and review["decision"]=="approve":
                    break
            node_id=f"N{turn:03d}"
            if strategy is not None and review["decision"]=="approve":
                measured,usage=run_exp(node_id,strategy,dev,"dev")
                outcome=audit(baseline["metrics"],measured["metrics"])
                score=outcome["merit"]
                status="MEASURED"
                if score>best["merit"] and outcome["eligible_development_only"]:
                    best_id=node_id
                else:
                    best_id=best["id"]
                node={"id":node_id,"parent":parent["id"],"depth":parent["depth"]+1,
                      "strategy":strategy,"status":status,"merit":score,
                      "search_priority":expansion_priority(baseline["metrics"],measured["metrics"]),
                      "expansions":0,
                      "tradeoff_tool_result":metric_tradeoff(baseline["metrics"],measured["metrics"]),
                      "metrics":measured["metrics"],"audit":outcome,
                      "requests":usage}
                nodes.append(node)
                seen.add(fingerprint(strategy))
                if best_id==node_id:
                    best=node
            else:
                outcome={"eligible_development_only":False,"reason":"CRITIC_OR_STRUCTURE_REJECTED"}
                nodes.append({"id":node_id,"parent":parent["id"],"depth":parent["depth"]+1,
                     "strategy":strategy,"status":"REJECTED","merit":-1,
                     "search_priority":-1.0,"expansions":0,
                     "metrics":None,"audit":outcome,"requests":0})
            reflection=ReflectionAgent(model).act({
                  "decision":proposals,"independent_development_audit":outcome,
                  "prior_dev_evidence":history[-2:],
                  "candidate_selected_so_far":best["id"],
                  "scientific_claim":"not independently confirmed"},
                  f"seed{seed}-node{turn}")
            record={"node":node_id,"parent":parent["id"],"diagnosis":diag,
                    "proposals":proposals,"audit":outcome,"reflection":reflection}
            history.append(record)
            # Reuse the mainline scientific_skills lifecycle. An Agent's
            # research insight is a proposed skill, never validated on its own.
            episode_dir=rd/f"episode-{turn:03d}"
            episode_dir.mkdir()
            atomic_json(episode_dir/"audit.json",{
                "task":"llmrank_bias","action":node_id,
                "panel_measurements":outcome,
                "reason":"Exploratory development: no independent confirmation",
                "independently_confirmed_discoveries":0})
            skill=record_skill(episode_dir,"llmrank_bias",
                     reflection["candidate_research_skill"],episode_dir/"audit.json")
            record["candidate_skill_id"]=skill["skill_id"]
            atomic_json(rd/"tree_index.json",{
                 "root":"ROOT","nodes":nodes,"iterations":history,"best":best["id"],
                 "independent_discoveries":0})
            atomic_json(rd/"scientific_memory.json",{
                "source":"numeric development audits, not prose conclusions",
                "recent_experiments":[{"id":n["id"],"status":n["status"],
                    "audit":n.get("audit")} for n in nodes[1:]],
                "critic_and_reflection":[{"node":h["node"],
                    "reasons":[x["review"]["reason"] for x in h["proposals"]],
                    "next_question":h["reflection"].get("next_question")}
                    for h in history[-3:]],
                "confirmation":"unseen holdout has not yet been opened"})
        tree_end=gateway.calls
        experiment_cost={
            "baseline_dev_requests":cost_base,
            "fixed_search_requests":fixed_end-fixed_start,
            "agent_tree_total_requests":tree_end-tree_start,
            "matched_node_slots":max_nodes,
            "strict_equal_compute":False,
        }
        # Only the FINAL independent confirmation run may open hidden test
        # labels. Intermediary RSI rounds end with development-only evidence.
        if development_only:
            result={
                "cohort":protocol,
                "development":{"baseline":baseline["metrics"],"fixed":fixed,
                    "selected_fixed":best_fixed,"selected_rsi":best,"tree":nodes},
                "holdout":None,"ci95":None,
                "evidence":{"confirmed_discoveries":0,
                    "phase":"development_only","no_holdout_opened":True,
                    "development_candidate_success":sum(
                        n.get("audit",{}).get("eligible_development_only",False)
                        for n in nodes[1:])},
                "usage":{"cumulative_model_calls":gateway.calls,
                          **experiment_cost,
                          "ranking_cache_hits":gateway.cache_hits,
                          "preloaded_prior_rankings":gateway.preloaded_rankings,
                          "equal_realized_compute_not_guaranteed":True}
            }
            atomic_json(rd/"result.json",result)
            summaries.append({"seed":seed,"chosen_rsi":best["id"],
                              "selected_merit":best["merit"],
                              "development_candidates":result["evidence"]["development_candidate_success"]})
            atomic_json(output/"summary.json",{"seeds_done":summaries,
                        "remaining_seeds":seeds[len(summaries):],
                        "requests":gateway.calls,"authoritative_research_discoveries":0,
                        "holdout_opened":False})
            continue
        # Keep/discard is frozen on development; only now open held-out test.
        base_test,_=run_exp("plain",BASELINE,test,"holdout")
        fixed_test,_=run_exp("fixed_selected",best_fixed["strategy"],test,"holdout")
        rsi_test,_=run_exp("rsi_selected",best["strategy"],test,"holdout")
        pair_fixed=bootstrap_paired(base_test,fixed_test,pop,titles,seed=seed)
        pair_rsi=bootstrap_paired(base_test,rsi_test,pop,titles,seed=seed+1)
        pair_rsi_fixed=bootstrap_paired(fixed_test,rsi_test,pop,titles,seed=seed+2)
        result={
            "cohort":protocol,
            "development":{"baseline":baseline["metrics"],"fixed":fixed,
                "selected_fixed":best_fixed,"selected_rsi":best,"tree":nodes},
            "holdout":{"baseline":base_test["metrics"],"fixed":fixed_test["metrics"],
                       "rsi":rsi_test["metrics"]},
            "ci95":{"fixed_minus_baseline":pair_fixed,
                    "rsi_minus_baseline":pair_rsi,
                    "rsi_minus_fixed":pair_rsi_fixed},
            "evidence":{"confirmed_discoveries":0,
                "development_candidate_success":sum(
                    n.get("audit",{}).get("eligible_development_only",False)
                    for n in nodes[1:]),
                "authoritative_evidence":authoritative_evidence({}),
                "statement":"Only preselected development candidates assessed on untouched holdout"},
            "usage":{"cumulative_model_calls":gateway.calls,
                     **experiment_cost,
                     "model_details":model_description(config),
                     "matched_candidate_slots":max_nodes,
                     "equal_realized_compute_not_guaranteed":True,
                     "ranking_cache_hits":gateway.cache_hits},
        }
        atomic_json(rd/"result.json",result)
        summaries.append({"seed":seed,"chosen_rsi":best["id"],
            "chosen_fixed":best_fixed["name"],
            "baseline_NDCG5":base_test["metrics"]["5"]["NDCG@5"],
            "fixed_NDCG5":fixed_test["metrics"]["5"]["NDCG@5"],
            "rsi_NDCG5":rsi_test["metrics"]["5"]["NDCG@5"],
            "development_candidates":result["evidence"]["development_candidate_success"]})
        with (output/"results.tsv").open("a",encoding="utf-8") as tsv:
            if tsv.tell()==0:
                tsv.write("seed\tarm\tHR5\tNDCG5\tARP5\tAPLT5\tselected_development_node\n")
            for arm in ("baseline","fixed","rsi"):
                m=result["holdout"][arm]["5"]
                tsv.write(f"{seed}\t{arm}\t{m['HR@5']:.6f}\t{m['NDCG@5']:.6f}\t"
                          f"{m['ARP@5']}\t{m['APLT@5']}\t{best['id']}\n")
        atomic_json(output/"summary.json",{"seeds_done":summaries,
            "remaining_seeds":seeds[len(summaries):],"requests":gateway.calls,
            "authoritative_research_discoveries":0})
    atomic_json(output/"usage.json",usage_summary(output))
    return {"status":"completed","results":summaries,"requests":gateway.calls}
