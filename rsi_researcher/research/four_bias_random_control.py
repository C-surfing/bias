"""Add a frozen, label-blind random-strategy control to LLMRank science-tree.

Random strategy is sampled *before seeing candidate performance*. It preserves
the Qwen prompt and changes only numeric ranking correction, so cached frozen
model rankings are reused. Costs reported separately, not equalized artificially.
"""
import argparse
import json
import random
from pathlib import Path
from rsi_researcher.llmrank.dataset import load_dataset
from rsi_researcher.llmrank.strategy import BASELINE,canonical,evaluate_strategy
from rsi_researcher.llmrank.agents import RankGateway
from rsi_researcher.llmrank.science import audit
from rsi_researcher.llmrank.bias_objectives import bootstrap_target_bias
from rsi_researcher.llmrank.evidence import paired_clean_evidence
from rsi_researcher.runtime.models import create_model
from rsi_researcher.runtime.client import atomic_json,usage_summary
from rsi_researcher.runtime.limits import RunLimits

def compare(run,seed,data,dataset,trials=1):
    run=Path(run)
    seed_dir=run/f"seed-{seed}"
    assert (seed_dir/"result.json").is_file(),"science tree has not completed"
    assert json.loads((run/"program.json").read_text())["development_only"]
    origin=json.loads((seed_dir/"dev_plain.json").read_text())
    cohort=json.loads((seed_dir/"frozen_evaluator_only.json").read_text())["dev"]
    train,targets,known,titles,pop=load_dataset(Path(data),dataset)
    import json as _json
    config=_json.loads(Path('/data/cjh/rsi/project/configs/llmrank_qwen.json').read_text())
    rng=random.Random(seed+2017)
    path=run/"independent_random_control"
    path.mkdir(exist_ok=False)
    modeldir=path/"model";modeldir.mkdir()
    limits=RunLimits(modeldir,max_requests=trials*len(cohort)*2+8,max_tokens=900000,max_seconds=3600)
    model=create_model(modeldir,limits.max_requests,limits,config)
    preload=[run/"rank_cache.jsonl"]
    parent=json.loads((run/"program.json").read_text()).get("prior_campaign")
    visited=set()
    while parent:
        anc=Path(parent).resolve()
        if anc in visited: raise ValueError("cyclic campaign lineage")
        visited.add(anc)
        preload.append(anc/"rank_cache.jsonl")
        parent=json.loads((anc/"program.json").read_text()).get("prior_campaign")
    # Do not merge the aborted duplicate-inference cache: some entries have
    # identical prompt hashes with different model responses. Preserve it
    # separately as a determinism diagnostic, never overwrite frozen evidence.
    gateway=RankGateway(model,cache_path=path/"rank_cache.jsonl",preload_paths=tuple(preload))
    baseline={
        "metrics":origin["metrics"],
        "random":origin["rows_random"],
        "first":origin["rows_first"]}
    results=[]
    for trial in range(trials):
        strength=round(0.2+rng.random()*1.7,4)
        policy=rng.choice(["global","personalized_gate","protect_top3"])
        strategy=canonical({**BASELINE,"popularity_lambda":strength,"rerank_policy":policy})
        start=gateway.calls
        measured=evaluate_strategy(gateway,cohort,titles,pop,strategy,seed,
                    "precommitted_random_strategy_"+str(trial))
        evidence=paired_clean_evidence(baseline,measured,pop,titles)
        bias_target=json.loads((run/"program.json").read_text())["bias_target"]
        outcome=audit(baseline["metrics"],measured["metrics"],
                clean_evidence=evidence,bias_target=bias_target)
        bias_ci=bootstrap_target_bias(baseline,measured,pop,titles,bias_target,
                                      seed=seed+trial+321,draws=400)
        results.append({"trial":trial,"strategy":strategy,"metrics":measured["metrics"],
              "audit":outcome,"paired_target_bias_ci":bias_ci,
              "qwen_requests":gateway.calls-start,
              "experimental_budget_note":"numeric policy stage, original Qwen prompts unchanged"})
    result={"status":"COMPLETED_FROZEN_RANDOM_SEARCH_DEVELOPMENT",
       "seed":seed,"bias_target":bias_target,"dataset":dataset,
       "n_trials":trials,"random_seed":seed+2017,
       "random_suggestion_uses_labels":False,
       "baseline":origin["metrics"],
       "fixed":json.loads((seed_dir/"result.json").read_text())["development"]["fixed"],
       "agent_tree":json.loads((seed_dir/"result.json").read_text())["development"]["tree"],
       "random_search":results,
       "random_qwen_requests":gateway.calls,"qwen_cache_hits":gateway.cache_hits,
       "agent_and_fixed_costs":json.loads((seed_dir/"result.json").read_text())["usage"],
       "strict_equal_realized_model_compute":False,
       "independent_holdout_opened":False,
       "confirmed_discoveries":0}
    atomic_json(path/"result.json",result)
    atomic_json(path/"usage.json",(usage_summary(modeldir) if (modeldir/"calls.jsonl").is_file() else {"requests":0,"cached_ranking_calls_only":True}))
    print("RANDOM_CONTROL_RESULT",json.dumps({
        "seed":seed,"bias_target":bias_target,"model_requests":gateway.calls,
        "trials":[{"lambda":x["strategy"]["popularity_lambda"],
          "policy":x["strategy"]["rerank_policy"],
          "PopREO":x["metrics"]["5"].get("PopREO@5"),
          "NDCG":x["metrics"]["5"]["NDCG@5"],
          "PositionBias":x["metrics"]["5"].get("CandidatePositionBias_NDCG@5"),
          "eligible":x["audit"]["eligible_development_only"]}
                  for x in results]}),flush=True)
    return result
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--trials",type=int,default=1)
    a=p.parse_args()
    compare(a.run,a.seed,a.data,a.dataset,a.trials)
if __name__=="__main__": main()
