"""Deterministic, label-blind *programs* for zero-shot recommendation debias.

The Agent writes a JSON sequence of bounded reranking stages. Unlike choosing
one action from a menu, stages can compose nonlinear user-conditioned scoring,
diversity-aware greedy selection, and a configurable relevance-preserving prefix.
No Python execution, target access, filesystem reads or hidden labels.
"""
from __future__ import annotations
import ast
import json
import math
import re

from .formula import score as evaluate_score, validate as validate_score

PROGRAM_VERSION="debias_policy_program_v1"
MAX_STAGES=5

def normalize_program(raw):
    """Translate unambiguous generated stage notation to one canonical AST."""
    if raw in ("",None,"[]"):
        return []
    if not isinstance(raw,str) or len(raw)>1600:
        raise ValueError("policy_program must be <=1600 characters of JSON/literal")
    try:
        stages=json.loads(raw)
    except (TypeError,json.JSONDecodeError):
        try:
            stages=ast.literal_eval(raw)
        except (ValueError,SyntaxError,TypeError,RecursionError) as exc:
            raise ValueError("policy_program must contain literal stages") from exc
    if not isinstance(stages,list) or not 1<=len(stages)<=MAX_STAGES:
        raise ValueError("policy_program requires 1..5 stages")
    legal={"score","mmr","protect","head_budget"}
    out=[]
    for element in stages:
        if not isinstance(element,dict):
            raise ValueError("each stage must be a dictionary")
        if "op" in element:
            out.append(element)
            continue
        # Common Qwen equivalent: {"score":{"op":"add","expr":"..."}},
        # or multiple stage-key entries in a single dict.
        if not element or any(k not in legal for k in element):
            raise ValueError("unrecognized nested policy stage")
        for name,payload in element.items():
            if not isinstance(payload,dict):
                raise ValueError("nested policy stage needs a dictionary")
            marker=payload.get("op")
            # These aliases are semantically unambiguous in the containing
            # stage; unknown operations still fail compilation.
            acceptable={None,name,"add","replace","top"}
            if name=="protect":
                acceptable|={"keep_top","lock_top","freeze_top"}
            if marker not in acceptable:
                raise ValueError("ambiguous nested policy instruction")
            out.append({"op":name,**{k:v for k,v in payload.items() if k!="op"}})
    if not 1<=len(out)<=MAX_STAGES:
        raise ValueError("normalized policy has too many stages")
    return out

def compile_program(raw):
    steps=normalize_program(raw)
    if not steps:
        return ()
    compiled=[]
    for step in steps:
        if not isinstance(step,dict) or "op" not in step:
            raise ValueError("each program stage requires an op")
        op=step["op"]
        if op=="score":
            if set(step)!={"op","expr"}:
                raise ValueError("score stage only allows op, expr")
            compiled.append((op,validate_score(step["expr"])))
        elif op=="mmr":
            if set(step)!={"op","relevance_weight"}:
                raise ValueError("mmr stage only allows op, relevance_weight")
            weight=step["relevance_weight"]
            if isinstance(weight,bool) or not isinstance(weight,(float,int)) or not 0<=weight<=1:
                raise ValueError("mmr relevance_weight must be in [0,1]")
            compiled.append((op,float(weight)))
        elif op=="protect":
            if set(step)!={"op","top"} or type(step["top"]) is not int or not 0<=step["top"]<=5:
                raise ValueError("protect requires a prefix size 0..5")
            compiled.append((op,step["top"]))
        elif op=="head_budget":
            if set(step)!={"op","top","max_head"}:
                raise ValueError("head_budget requires top and max_head")
            k,m=step["top"],step["max_head"]
            if type(k) is not int or type(m) is not int or not 1<=k<=10 or not 0<=m<=k:
                raise ValueError("invalid bounded head_budget")
            compiled.append((op,(k,m)))
        else:
            raise ValueError("unknown program stage: "+str(op))
    return tuple(compiled)

def title_tokens(title):
    return set(re.findall(r"[a-z0-9]{3,}",str(title).lower()))

def jaccard(a,b):
    return len(a&b)/len(a|b) if a|b else 0.0

def execute_program(compiled, original, features, titles):
    """Apply an Agent-written policy to one cohort example, without labels.

    All stage values come from fixed candidate titles, frozen train popularity,
    user's observed training history and Qwen rank votes. No target information.
    """
    if not compiled:
        return original
    items=list(original)
    relevance={item:features[item]["score"] for item in items}
    anchor=items.copy()
    word_sets={item:title_tokens(titles[item]) for item in items}
    for op,arg in compiled:
        if op=="score":
            relevance={i:evaluate_score(arg,{**features[i],"score":relevance[i]})
                       for i in items}
            items.sort(key=lambda i:(-relevance[i],i))
        elif op=="mmr":
            remaining=set(items)
            selected=[]
            vmin=min(relevance.values())
            vmax=max(relevance.values())
            span=max(vmax-vmin,1e-9)
            while remaining:
                def mmr_value(i):
                    redundancy=max((jaccard(word_sets[i],word_sets[j]) for j in selected),
                                   default=0.0)
                    value=(relevance[i]-vmin)/span
                    return arg*value-(1-arg)*redundancy
                chosen=max(remaining,key=lambda i:(mmr_value(i),i))
                selected.append(chosen)
                remaining.remove(chosen)
            items=selected
        elif op=="protect":
            protected=anchor[:arg]
            fixed=set(protected)
            items=protected+[i for i in items if i not in fixed]
        elif op=="head_budget":
            top_k,max_head=arg
            high=sum(1 for i in items if features[i]["head"]>0.5 and i in items[:top_k])
            if high>max_head:
                # The best-scored nonhead replaces the lowest-ranked excess
                # head inside top-k, subject to candidate inventory.
                candidate=[i for i in items[top_k:] if features[i]["head"]<=0.5]
                for newcomer in candidate:
                    if high<=max_head:
                        break
                    positions=[j for j in range(top_k)
                               if features[items[j]]["head"]>0.5]
                    if not positions:
                        break
                    old=items[positions[-1]]
                    items.remove(newcomer)
                    items.insert(positions[-1],newcomer)
                    items.remove(old)
                    items.insert(top_k,old)
                    high-=1
        if len(items)!=len(anchor) or set(items)!=set(anchor):
            raise RuntimeError("debias program changed frozen candidate inventory")
    return items
