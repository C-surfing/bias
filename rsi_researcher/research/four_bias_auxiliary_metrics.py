"""Reviewer-only auxiliary metrics for measured LLMRank RSI arms.

Uses frozen development rows, training user histories and catalog popularity.
Does not alter primary scientific search scores or expose holdout labels.
"""
import argparse
import json
import math
from pathlib import Path
from rsi_researcher.llmrank.dataset import _catalog_groups,load_dataset
from rsi_researcher.runtime.client import atomic_json

def js_div(p,q):
    mid=[(a+b)*.5 for a,b in zip(p,q)]
    def div(a):
        return sum(x*math.log2(x/y) for x,y in zip(a,mid) if x>0)
    return .5*(div(p)+div(q))
def group_dist(items,group):
    counts=[0]*5
    for item in items:
        if item in group:counts[group[item]]+=1
    tot=sum(counts)
    return [x/tot for x in counts] if tot else None

def auxiliary(dataset,run,seed,datadir):
    directory=Path(run)
    archive=directory/f"seed-{seed}"
    assert json.loads((directory/"program.json").read_text())["development_only"]
    cohort=json.loads((archive/"frozen_evaluator_only.json").read_text())["dev"]
    hist={c["user"]:c["history"] for c in cohort}
    _,_,_,catalog,pop=load_dataset(Path(datadir),dataset)
    _,group,_=_catalog_groups(pop,catalog)
    result={}
    for path in sorted(archive.glob("dev_*.json")):
        blob=json.loads(path.read_text())
        if "rows_random" not in blob or "rows_first" not in blob:continue
        r=blob["rows_random"];f=blob["rows_first"]
        assert len(r)==len(f)==len(hist)
        calibrations=[]
        overlaps=[]
        clean_overlaps=[]
        for a,b in zip(r,f):
            assert a["user"]==b["user"]
            user=a["user"]
            pd=group_dist(hist[user],group)
            rank_a=a["ranking"]
            rank_b=b["ranking"]
            if pd and rank_a:
                qd=group_dist(rank_a[:5],group)
                if qd: calibrations.append(js_div(pd,qd))
            if rank_a and rank_b:
                val=len(set(rank_a[:5]) & set(rank_b[:5]))/5
                overlaps.append(val)
                if not a.get("repaired_votes") and not b.get("repaired_votes"):
                    clean_overlaps.append(val)
        result[path.stem]={
            "PopularityCalibrationJS@5":sum(calibrations)/len(calibrations) if calibrations else None,
            "Top5SetOverlapFirstVsRandom":sum(overlaps)/len(overlaps) if overlaps else None,
            "Top5SetOverlapCleanOnly":sum(clean_overlaps)/len(clean_overlaps) if clean_overlaps else None,
            "calibration_supported_users":len(calibrations),
            "position_valid_pairs":len(overlaps),
            "position_clean_pairs":len(clean_overlaps),
            "main_metrics_unchanged":True}
    atomic_json(directory/"reviewer_auxiliary_metrics.json",result)
    print("AUXILIARY_METRICS",json.dumps(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--seed",type=int,required=True)
    a=p.parse_args();auxiliary(a.dataset,a.run,a.seed,a.data)
