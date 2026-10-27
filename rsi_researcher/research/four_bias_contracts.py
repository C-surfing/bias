"""Four bias task evidence contracts. Scientific promotion is evaluator-only."""
from dataclasses import dataclass
from math import isfinite
@dataclass(frozen=True)
class BiasContract:
    name:str
    metric:str
    direction:str
    identification:str
    utility:tuple
    min_gain:float
    required_evidence:tuple

CONTRACTS={
 "popularity":BiasContract("popularity","PopREO@5","min","observed_positive_reranking",("NDCG@5","HR@5"),.02,("matched_candidates","group_support","clean_parser")),
 "candidate_position":BiasContract("candidate_position","CandidatePositionBias_NDCG@5","abs_min","paired_prompt_order",("NDCG@5","HR@5"),.02,("same_candidates","paired_permutations","clean_parser")),
 "selection":BiasContract("selection","Brier","min","known_extra_MNAR_mask_on_near_full_KuaiRec",("NDCG@50",),.001,("reference_panel","nonzero_observation_probability")),
 "exposure":BiasContract("exposure","SNIPS","max","known_random_logger_propensity",("SNIPS",),.0001,("propensity","support","ESS"))
}
def assess(target,before,after,*,identification,evidence_grade,ess=None,relative_tolerance=.05):
    c=CONTRACTS[target]
    def get(m,key):
        v=m.get(key)
        return abs(v) if c.direction=="abs_min" and v is not None else v
    a,b=get(before,c.metric),get(after,c.metric)
    defined=all(type(x) in (float,int) and isfinite(x) for x in (a,b))
    delta=(a-b if c.direction!="max" else b-a) if defined else None
    good=defined
    differences={}
    if target!="exposure":
        for name in c.utility:
            x,y=before.get(name),after.get(name)
            differences[name]=y-x if x is not None and y is not None else None
            good=good and x is not None and y is not None and y>=x*(1-relative_tolerance)
    else:
        good=good and ess is not None and ess>=250
    identification_ok=identification==c.identification
    eligible=bool(good and identification_ok and delta>c.min_gain)
    return {"target":target,"primary":c.metric,"bias_reduction":delta,
        "utility_deltas":differences,"support_valid":bool(good),
        "identification_ok":identification_ok,"evidence_grade":evidence_grade,
        "ess":ess,"development_eligible":eligible,
        "status":"not_identified" if not identification_ok else
                 "insufficient_support_or_utility_loss" if not good else
                 "development_eligible_not_confirmed" if eligible else "target_not_improved",
        "confirmed_discoveries":0,"holdout_opened":False}
