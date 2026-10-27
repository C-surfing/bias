"""Executable, compositional (not enumerated-action) LLMRank debias strategy DSL.

Agents may invent evidence-based *instructions* and combine independently
implemented mechanisms. Python validates, compiles, runs and measures them.
Only development metrics are visible to scientific agents.
"""
from __future__ import annotations
from collections import Counter
from .formula import validate as validate_formula, score as evaluate_formula
from .program import compile_program, normalize_program, execute_program, PROGRAM_VERSION
import hashlib
import json
import math
import random

PARSER_CONTRACT="rank_alias_v2_open_policy_program_v1_history_count"

SPEC = {
    "history_n": (3,5,10,20,50),
    "history_order": ("chronological","reverse"),
    "candidate_order": ("frozen","reversed","rotated"),
    "votes": (1,3),
    # This is no longer an enumerated choice: continuous strength [0,2].
    "popularity_lambda": None,
    "rerank_policy": ("global","personalized_gate","protect_top3"),
}
BASELINE = {
    "history_n":20,"history_order":"chronological","candidate_order":"frozen",
    "votes":1,"popularity_lambda":0,"instruction":"","score_expression":"base",
    "rerank_policy":"global",
    "policy_program":"",
}
FIXED = {
    "fixed_popularity":{**BASELINE,"popularity_lambda":0.1},
    "fixed_personalized":{**BASELINE,"popularity_lambda":0.2,
                          "rerank_policy":"personalized_gate"},
    "fixed_protected":{**BASELINE,"popularity_lambda":0.2,
                        "rerank_policy":"protect_top3"},
    "fixed_recency":{**BASELINE,"history_n":5,"instruction":"Prioritize the most recent interactions."},
    "fixed_balance":{**BASELINE,"instruction":"Focus on personalized fit over global item popularity."},
    "fixed_bootstrap":{**BASELINE,"votes":3},
}

def canonical(input):
    if not isinstance(input,dict):
        raise ValueError("strategy must be a JSON object")
    # An omitted new formula field falls back to neutral LLM rank votes.
    # Older Qwen responses remain executable; never default other semantics.
    mandatory=set(BASELINE)-{"score_expression","rerank_policy","policy_program"}
    if not mandatory <= set(input) or not set(input) <= set(BASELINE):
        raise ValueError(f"required strategy fields: {sorted(mandatory)}")
    result={}
    for key, allowed in SPEC.items():
        v=input.get(key,BASELINE[key])
        if key=="popularity_lambda":
            if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=2:
                raise ValueError("popularity_lambda must be finite continuous number 0..2")
            result[key]=float(v)
        else:
            if isinstance(v,bool) or v not in allowed:
                raise ValueError(f"invalid {key}: {v}")
            result[key]=v
    instruction=input["instruction"]
    if not isinstance(instruction,str) or len(instruction)>350 or any(c in instruction for c in "\r\x00"):
        raise ValueError("instruction must be <=350 characters of plain text")
    result["instruction"]=instruction.strip()
    expression=input.get("score_expression","base")
    validate_formula(expression)
    result["score_expression"]=expression
    program=input.get("policy_program","")
    compiled=compile_program(program)
    result["policy_program"]=(json.dumps(normalize_program(program),
                   sort_keys=True,separators=(",",":")) if compiled else "")
    return result

def fingerprint(strategy):
    return hashlib.sha256(json.dumps(strategy,sort_keys=True).encode()).hexdigest()[:12]

def functional_fingerprint(strategy,cases):
    """Deduplicate parameter variants that cause identical cohort behavior.

    For a cohort with max 19 available history items, a 20-item and a
    50-item history window are equivalent. No-op popularity policies are
    also collapsed when the score formula is neutral and penalty is zero.
    This prunes provable equivalences, NOT entire unsuccessful bias families.
    """
    normalized=canonical(strategy)
    max_h=max((len(case["history"]) for case in cases),default=0)
    normalized["history_n"]=min(normalized["history_n"],max_h)
    normalized["score_expression"]=" ".join(normalized["score_expression"].split())
    if (normalized["popularity_lambda"]==0 and
            normalized["score_expression"]=="base"):
        normalized["rerank_policy"]="global"
    return fingerprint(normalized)

def stable_random(seed, *parts):
    digest=hashlib.sha256(("/".join(map(str,(seed,*parts)))).encode()).digest()
    return random.Random(int.from_bytes(digest[:8],"big"))

def reorder(candidates, strategy, seed, user, replicate):
    result=candidates.copy()
    method=strategy["candidate_order"]
    if method=="reversed":
        result.reverse()
    elif method=="rotated":
        start=(seed+replicate) % len(result)
        result=result[start:]+result[:start]
    if replicate:
        stable_random(seed,user,replicate).shuffle(result)
    return result

def stable_aliases(case):
    """Per-user deterministic random aliases, unchanged by position arm."""
    items=sorted(case["candidates"])
    stable_random(0,case["user"],*items).shuffle(items)
    return {item:f"C{idx+1:02d}" for idx,item in enumerate(items)}

def make_prompt(case, order, titles, strategy):
    history=case["history"][-strategy["history_n"]:]
    if strategy["history_order"]=="reverse":
        history=history[::-1]
        heading="User interactions (most recent first):"
    else:
        heading="User interactions (oldest to newest):"
    hist="\n".join(f"- {titles[i]}" for i in history if i in titles)
    aliases=stable_aliases(case)
    opts="\n".join(f"- {aliases[item]}: {titles[item]}" for item in order)
    guidance=strategy["instruction"] or "Rank the candidates by user-specific relevance."
    return (heading+"\n"+hist+"\nCandidates:\n"+opts+
            "\n"+guidance+
            "\nReturn only one JSON object with key ranking containing ALL candidate "+
            "Cxx aliases exactly once, ordered most to least relevant; "+
            "never output ordinal positions, titles or unseen identifiers."),aliases

def repair_alias_permutation(raw,expected):
    """Label-blind deterministic repair of a mostly complete LLM ranking.

    Keep first occurrence of every valid alias; fill omissions using the
    user's stable pseudo-random aliases, not display position or target label.
    Require at least half the requested candidates to be recognizable.
    The unmodified model response remains in the immutable Qwen call log.
    """
    if not isinstance(raw,list):
        return None,False
    size=len(expected)
    unique=[]
    seen=set()
    for alias in raw:
        if isinstance(alias,str) and alias in expected and alias not in seen:
            seen.add(alias)
            unique.append(alias)
    if len(unique)<max(3,size//2):
        return None,False
    repaired=(len(unique)!=size or len(raw)!=size)
    unique.extend(alias for alias in sorted(expected) if alias not in seen)
    if len(unique)!=size or len(set(unique))!=size:
        raise RuntimeError("internal permutation reconstruction contract violated")
    return unique,repaired

def rank(model,case,order,titles,strategy,popularity,seed,tag,
         max_requests=None,return_details=False):
    scores=Counter()
    n=len(case["candidates"])
    alias_expected={alias:item for item,alias in stable_aliases(case).items()}
    repaired_votes=0
    observed_votes=0
    policy_failures=0
    def return_result(prediction):
        meta={"repaired_votes":repaired_votes,"observed_votes":observed_votes,
              "policy_execution_failures":policy_failures}
        return (prediction,meta) if return_details else prediction
    for repeat in range(strategy["votes"]):
        if max_requests is not None and model.calls>=max_requests:
            raise RuntimeError("total Qwen request budget exhausted")
        ordered=reorder(order,strategy,seed,case["user"],repeat)
        prompt, aliases=make_prompt(case,ordered,titles,strategy)
        try:
            result=model.ask("You are a zero-shot LLMRank-style personalized ranking system. Return JSON only.",
                             prompt,f"{tag}.{repeat}")
            observed_votes+=1
            out,was_repaired=repair_alias_permutation(result["ranking"],alias_expected)
            if out is None:
                return return_result(None)
            repaired_votes+=int(was_repaired)
            for position,alias in enumerate(out):
                scores[alias_expected[alias]]+=n-position
        except (ValueError,KeyError,TypeError):
            return return_result(None)
    # A deterministic, independent post-ranking correction. Does NOT see gt.
    lam=strategy["popularity_lambda"]
    popmax=max((popularity.get(i,0) for i in case["candidates"]),default=0)
    scale=math.log1p(popmax) or 1.0
    tree=validate_formula(strategy["score_expression"])
    compiled=compile_program(strategy.get("policy_program",""))
    import re
    def words(text):
        return set(re.findall(r"[a-z0-9]{3,}",text.lower()))
    hist=set().union(*(words(titles[i]) for i in case["history"][-strategy["history_n"]:] if i in titles))
    # Original rank and user profile come strictly from training evidence.
    original=sorted(case["candidates"],key=lambda i:(-scores[i],i))
    original_position={item:idx for idx,item in enumerate(original)}
    pop_order=sorted(case["candidates"],key=lambda i:(popularity.get(i,0),i))
    pop_quantile={item:idx/max(1,n-1) for idx,item in enumerate(pop_order)}
    head_items=set(pop_order[max(0,n-math.ceil(.2*n)):])
    recent_hist=set().union(*(words(titles[i]) for i in case["history"][-3:] if i in titles))
    policy=strategy.get("rerank_policy","global")
    history_pop=[popularity.get(i,0) for i in case["history"] if i in popularity]
    hist_cutoff=sorted(history_pop)[len(history_pop)//2] if history_pop else 0
    normalized_cutoff=min(1.,math.log1p(hist_cutoff)/scale)
    features_by_item={}
    for item in case["candidates"]:
        normalized=math.log1p(popularity.get(item,0))/scale
        tokens=words(titles[item])
        overlap=len(tokens&hist)/(len(tokens) or 1)
        base=scores[item]/(n*strategy["votes"])
        features={"base":base,"score":base,"pop":normalized,
            "match":overlap,
            "recent_match":len(tokens&recent_hist)/(len(tokens) or 1),
            "user_pop":normalized_cutoff,
            "pop_excess":max(0,normalized-normalized_cutoff),
            "novelty":1.-normalized,
            "head":float(item in head_items),
            "rank_norm":1.-original_position[item]/max(1,n-1),
            "history_len":float(len(case["history"])),
            "pop_percentile":pop_quantile[item]}
        penalty=lam*(features["pop_excess"]
                     if policy=="personalized_gate" else normalized)
        try:
            final_score=evaluate_formula(tree,features)-penalty
        except (ValueError,ArithmeticError):
            policy_failures+=1
            return return_result(None)
        scores[item]=final_score
        features["score"]=final_score
        features_by_item[item]=features
    proposed=sorted(case["candidates"],key=lambda i:(-scores[i],i))
    if policy=="protect_top3":
        keep=set(original[:3])
        proposed=original[:3]+[item for item in proposed if item not in keep]
    if compiled:
        try:
            proposed=execute_program(compiled,proposed,features_by_item,titles)
        except (ValueError,ArithmeticError):
            policy_failures+=1
            return return_result(None)
    return return_result(proposed)

def evaluate_strategy(model,cases,titles,popularity,strategy,seed,label,max_requests=None):
    from .dataset import summarize, cpb
    cases=tuple(cases)
    random_rows=[]
    first_rows=[]
    for index,case in enumerate(cases):
        shuffled=case["candidates"].copy()
        first=[case["target"]]+[i for i in shuffled if i!=case["target"]]
        for name,order,rows in (("random",shuffled,random_rows),("first",first,first_rows)):
            prediction,quality=rank(model,case,order,titles,strategy,popularity,
                            seed+index*101,f"{label}.{index}.{name}",max_requests,
                            return_details=True)
            rows.append({"user":case["user"],"target":case["target"],
                         "ranking":prediction,**quality})
    metrics={}
    for k in (5,10):
        x=summarize(random_rows,popularity,titles,k)
        y=summarize(first_rows,popularity,titles,k)
        x[f"CandidatePositionBias_HR@{k}"]=cpb(y[f"HR@{k}"],x[f"HR@{k}"])
        x[f"CandidatePositionBias_NDCG@{k}"]=cpb(y[f"NDCG@{k}"],x[f"NDCG@{k}"])
        metrics[str(k)]=x
    return {"metrics":metrics,"random":random_rows,"first":first_rows}

def bootstrap_paired(before,after,popularity,catalog,k=5,seed=20261021,draws=2000):
    """Paired user-resample CIs for utility, popularity and position proxies.

    The same user indices are sampled in baseline and intervention; position
    probes use each user's frozen random and target-first presentation arms.
    Undefined small-sample saturated CPB values stay unsupported.
    """
    from .dataset import summarize,cpb
    a,b=before["random"],after["random"]
    af,bf=before["first"],after["first"]
    n=len(a)
    if (n==0 or n!=len(b) or n!=len(af) or n!=len(bf) or
        any(x["user"]!=y["user"] for x,y in zip(a,b)) or
        any(x["user"]!=y["user"] for x,y in zip(a,af)) or
        any(x["user"]!=y["user"] for x,y in zip(a,bf))):
        raise ValueError("paired bootstrap requires matching users and both position arms")
    ordinary=(f"HR@{k}",f"NDCG@{k}",f"ARP@{k}",f"APLT@{k}")
    position=(f"CandidatePositionBias_HR@{k}",f"CandidatePositionBias_NDCG@{k}")
    keys=ordinary+position
    rng=random.Random(seed)
    samples={key:[] for key in keys}

    def observed(x,xf):
        rank=summarize(x,popularity,catalog,k)
        first=summarize(xf,popularity,catalog,k)
        for suffix in ("HR","NDCG"):
            name=f"{suffix}@{k}"
            rank[f"CandidatePositionBias_{name}"]=cpb(first[name],rank[name])
        return rank

    baseline=observed(a,af)
    changed=observed(b,bf)
    for _ in range(draws):
        index=[rng.randrange(n) for _ in range(n)]
        left=observed([a[i] for i in index],[af[i] for i in index])
        right=observed([b[i] for i in index],[bf[i] for i in index])
        for key in keys:
            if left.get(key) is not None and right.get(key) is not None:
                samples[key].append(right[key]-left[key])
    results={}
    for key in keys:
        x,y=baseline.get(key),changed.get(key)
        values=sorted(samples[key])
        supported=len(values)>=int(draws*0.8)
        results[key]={
            "delta":(y-x if x is not None and y is not None else None),
            "ci95":([values[int(.025*len(values))],
                     values[min(len(values)-1,int(.975*len(values)))]]
                    if supported and values else None),
            "paired_n":n,
            "valid_resamples":len(values),
            "requested_resamples":draws,
        }
    return results
