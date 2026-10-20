"""Five registered recommendation Bias tasks: real development methods and safe abstentions.

An EVIDENCE adapter, not another Agent / researcher. All task definitions live
in bias_registry; source data is read-only; no held-out labels enter metrics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from csv import DictReader
from hashlib import sha256
from math import log, log1p
from pathlib import Path
from statistics import mean
from random import Random
import argparse
import json

from rsi_researcher.research import bias_registry, exposure_synthetic
from rsi_researcher.research.coat_selection import (
    DATA, SEED, THRESHOLD, fit_item_mean, estimated_p_matrix, metrics,
)
from rsi_researcher.datasets.coat import load_coat
from rsi_researcher.evaluation.observed import evaluate_observed
from rsi_researcher.runtime.client import atomic_json

ROOT = Path(__file__).resolve().parents[2]
KUAIRAND = Path("/data/cjh/rsi/project/data/external/kuairand-pure/KuaiRand-Pure/data")
COAT_FEATURES = DATA / "user_item_features"
VERSION = "five-classic-bias-development-evidence-v1"


def _hash_bucket(*parts):
    return int.from_bytes(sha256("|".join(map(str, parts)).encode()).digest()[:8],
                          "big") % 10


def _click_model(rows, *, smoothing=20):
    """Smoothed item/tab click model fit exclusively on provided logged rows."""
    sums=defaultdict(int); counts=Counter()
    positives=0; n=0
    for key,click in rows:
        counts[key]+=1
        sums[key]+=click
        positives+=click
        n+=1
    if not n:
        raise ValueError("empty logged training input")
    prior=positives/n
    return {k:(sums[k]+smoothing*prior)/(count+smoothing)
            for k,count in counts.items()}, prior, counts


def exposure_real(random_path, standard_path):
    """Logged random-reference model comparison; NO individual OPE propensity.

    Predeclared hash: random 0-5 training, 6-7 development, 8-9 untouched.
    Standard is training only, from the same reported date interval.
    All models use the same item + UI-tab key, smoothing, and dev impressions.
    """
    random_path, standard_path=Path(random_path),Path(standard_path)
    standard=[]
    for path,rows in ((standard_path,standard),):
        with path.open(encoding="utf8",newline="") as stream:
            for row in DictReader(stream):
                if row["is_rand"]!="0":
                    raise ValueError("standard log contains randomized samples")
                rows.append(((row["video_id"],row["tab"]),int(row["is_click"])))
    train=[];dev=[];reserve=0
    with random_path.open(encoding="utf8",newline="") as stream:
        for row in DictReader(stream):
            if row["is_rand"]!="1":
                raise ValueError("random log contains nonrandom samples")
            bucket=_hash_bucket(row["user_id"],row["video_id"],row["time_ms"])
            if bucket<=5:
                train.append(((row["video_id"],row["tab"]),int(row["is_click"])))
            elif bucket<=7:
                dev.append((row["user_id"],(row["video_id"],row["tab"]),
                            int(row["is_click"])))
            else:
                reserve+=1  # Never access held-out outcome or fit on its labels.
    # Equal-number experiment: fixed seeded draw exclusively from the
    # randomized TRAIN partition, never from development or holdout.
    random_equal=Random(20261020).sample(train,len(standard))
    m_standard,p_standard,c_standard=_click_model(standard)
    m_random,p_random,c_random=_click_model(train)
    m_equal,p_equal,c_equal=_click_model(random_equal)
    if not dev or reserve==0:
        raise ValueError("not enough randomized support for reserved split")
    output={}
    # Common-support error cannot be driven by a model predicting from a
    # fallback global prior on keys absent from its training data.
    shared_keys=set(c_standard)&set(c_equal)
    if not shared_keys:
        raise ValueError("no item/tab common support between logging arms")
    for name,predictions,prior,training_count in (
            ("standard_only",m_standard,p_standard,len(standard)),
            ("random_reference",m_random,p_random,len(train)),
            ("random_equal_budget",m_equal,p_equal,len(random_equal))):
        panels={"A":[0,0.0],"B":[0,0.0]}
        common=[0,0.0]; uncovered=0; loss=0.0
        for u,key,click in dev:
            if key not in predictions:
                uncovered+=1
            pred=predictions.get(key,prior)
            err=(pred-click)**2
            loss+=err
            if key in shared_keys:
                common[0]+=1
                common[1]+=err
            panel="A" if _hash_bucket(u)<5 else "B"
            panels[panel][0]+=1
            panels[panel][1]+=err
        output[name]={
            "brier":loss/len(dev),
            "common_support_brier":common[1]/common[0] if common[0] else None,
            "common_support_development_impressions":common[0],
            "panel_brier":{k:(v[1]/v[0] if v[0] else None)
                           for k,v in panels.items()},
            "training_impressions":training_count,
            "unseen_item_tab_fraction":uncovered/len(dev),
            "prediction_rule":"training-only item/tab smoothed CTR, unseen -> train prior",
        }
    return {
        "status":"REAL_RANDOM_EXPOSURE_DEVELOPMENT_COMPARISON",
        "bias_id":"exposure",
        "intervention":"random_reference_click_training_vs_standard_click_training",
        "random_dev_impressions":len(dev),
        "random_reserved_impressions_unscored":reserve,
        "random_train_impressions":len(train),
        "standard_train_impressions":len(standard),
        "same_day_window":"2022-04-22 to 2022-05-08",
        "random_split":"stable user/video/time hash: 60% train, 20% development, 20% untouched",
        "equal_budget_random_sample_seed":20261020,
        "same_training_impressions_control":len(random_equal)==len(standard),
        "shared_item_tab_keys":len(shared_keys),
        "policies":output,
        "true_full_logging_propensities_known":False,
        "causal_OPE_identified":False,
        "independent_confirmation":False,
        "limits":[
            "KuaiRand-Pure retains a restricted candidate pool and incomplete histories",
            "Placement, UI tab and random insertion mechanisms differ from ordinary recommendations",
            "An equal-impressions arm controls sample count, not distinct-item coverage, "
            "UI placement, per-user selection or effective training cost",
            "This compares prediction under randomized intervention, not counterfactual policy value",
            "The reserved random partition is NEVER evaluated in this development run"],
    }


def _head_items(train, item_ids):
    counts=Counter(x["item"] for x in train)
    ranked=sorted(item_ids,key=lambda i:(-counts[i],i))
    return set(ranked[:max(1,len(ranked)//5)]),counts


def popularity_coat(ds):
    """Fixed candidate-popularity penalty, scored on SAME random Coat dev rows."""
    train,dev,items=ds["train"],ds["development"],ds["item_ids"]
    # Use a train-only estimated-propensity model for constant candidate scores.
    scores,_=fit_item_mean(train,items,estimated_p_matrix(),100)
    head,counts=_head_items(train,items)
    max_count=max(counts.values())
    by_user=defaultdict(list)
    for row in dev:by_user[row["user"]].append(row)
    variants={}
    for strength in (0.0,0.05,0.1,0.2):
        ranks={}
        for u,rows in by_user.items():
            ranks[u]=sorted((r["item"] for r in rows),
                key=lambda item:(-scores[item] +
                                  strength*log1p(counts[item])/log1p(max_count),item))
        base=evaluate_observed(dev,ranks,counts,catalog_items=items,k=5,
                               positive_threshold=THRESHOLD,gain_floor=1)
        denominators={p:0 for p in ("A","B")}
        numerators={p:0 for p in ("A","B")}
        for user,r in ranks.items():
            g="A" if int(user)%2==0 else "B"
            denominators[g]+=len(r[:5])
            numerators[g]+=sum(i in head for i in r[:5])
        variants[str(strength)]={
            "ndcg_at_5":base["ndcg"],
            "tail_relevant_recall":base["tail_relevant_recall"],
            "head_exposure_share":base["head_share"],
            "head_top_20pct_exposure_share":sum(
                numerators.values())/sum(denominators.values()),
            "head_share_panel":{p:numerators[p]/denominators[p] if denominators[p] else None
                                 for p in ("A","B")},
            "mean_quality_users":base["quality_users"],
            "intervention_strength":strength,
        }
    baseline=variants["0.0"]
    for key,row in variants.items():
        row["quality_guard_99pct_ndcg"]=row["ndcg_at_5"]>=.99*baseline["ndcg_at_5"]
        row["head_share_change_vs_baseline"]=(
            row["head_top_20pct_exposure_share"] -
            baseline["head_top_20pct_exposure_share"])
    return {
        "bias_id":"popularity",
        "status":"RANDOM_RATING_DEVELOPMENT_PROXY_BENCHMARK",
        "dataset":"Coat random-rating development, not AgentCF causal popularity",
        "intervention":"train-frequency log penalty on top-k item mean ranking",
        "development_users":len(by_user),
        "head_definition":"top 20% items by TRAIN-only nonrandom rating frequency",
        "fixed_strength_grid":[0.0,0.05,0.1,0.2],
        "policies":variants,
        "independent_relevance_reference":"randomly rated (partial candidate) Coat development",
        "identifies_unjustified_popularity_causally":False,
        "independent_confirmation":False,
        "limitations":"Fixed development candidate sets (usually small). Head exposure reduction "
                       "does not prove popularity bias or preserve unobserved full-catalog utility."
    }


def _coat_groups():
    names=(COAT_FEATURES/"user_features_map.txt").read_text(encoding="utf8").splitlines()
    assert names[:2]==["gender:men","gender:women"]
    groups={}
    with (COAT_FEATURES/"user_features.ascii").open(encoding="utf8") as stream:
        for i,line in enumerate(stream):
            bits=[float(s) for s in line.split()]
            if len(bits)!=len(names):
                raise ValueError("Coat user feature dimension mismatch")
            if bits[0]+bits[1]==1:
                groups[str(i)]="men" if bits[0] else "women"
    return groups


def fairness_coat(ds):
    """Train-only group calibration vs unadjusted item mean on randomized dev.

    Normative estimand: *error parity*, NOT equal opportunity or social justice.
    """
    groups=_coat_groups()
    train,dev,items=ds["train"],ds["development"],ds["item_ids"]
    base,_=fit_item_mean(train,items,estimated_p_matrix(),100)
    residuals=defaultdict(float);counts=Counter()
    for row in train:
        group=groups.get(row["user"])
        if group is None:continue
        residuals[group]+=int(row["rating"]>=THRESHOLD)-base[row["item"]]
        counts[group]+=1
    if any(counts[g]<30 for g in ("men","women")):
        return {"bias_id":"fairness","status":"ABSTAIN_INSUFFICIENT_GROUP_TRAINING"}
    # Fixed shrinkage; deliberately not tuned on randomized development labels.
    adjustment={g:residuals[g]/(counts[g]+100) for g in counts}
    def evaluate(adjusted):
        per={g:{"A":[0,0.0],"B":[0,0.0]} for g in ("men","women")}
        for row in dev:
            group=groups.get(row["user"])
            if group not in per:continue
            value=base[row["item"]]+(adjustment[group] if adjusted else 0)
            value=min(1,max(0,value))
            sq=(value-int(row["rating"]>=THRESHOLD))**2
            panel="A" if int(row["user"])%2==0 else "B"
            per[group][panel][0]+=1
            per[group][panel][1]+=sq
        summary={}
        for group,panel in per.items():
            n=sum(x[0] for x in panel.values())
            summary[group]={
                "development_ratings":n,
                "brier":sum(x[1] for x in panel.values())/n if n else None,
                "panel_brier":{key:(n1[1]/n1[0] if n1[0] else None)
                               for key,n1 in panel.items()},
            }
        return summary
    before=evaluate(False);after=evaluate(True)
    if any(x["development_ratings"]<30 for x in before.values()):
        status="ABSTAIN_INSUFFICIENT_RANDOM_REFERENCE_PER_GROUP"
    else:
        status="GROUP_PREDICTION_CALIBRATION_DEVELOPMENT"
    gap=lambda x:abs(x["men"]["brier"]-x["women"]["brier"]) if (
        x["men"]["brier"] is not None and x["women"]["brier"] is not None) else None
    return {
        "bias_id":"fairness","status":status,
        "group_definition":"Coat user_features_map gender:men vs gender:women",
        "criterion":"groupwise Brier error disparity on randomized-rated items",
        "normative_scope":"predictive error parity only, not group opportunity parity",
        "intervention":"training-only group residual intercept, shrinkage=100",
        "train_group_rows":dict(counts),
        "fitted_intercepts":adjustment,
        "unadjusted":before,"group_calibrated":after,
        "absolute_brier_gap_before":gap(before),
        "absolute_brier_gap_after":gap(after),
        "recommendation_rank_change":False,
        "independent_confirmation":False,
        "limits":["Self-selected training group residuals may be MNAR",
                  "Group labels from one dataset do not define universal fairness",
                  "Offset adjustment changes calibration, not within-user ranking",
                  "No unobserved outcomes, sensitive group benefits or normative fairness inferred"],
    }


def filter_bubble_kuairand(feature_path, early_path, late_path):
    """Longitudinal observational SUPPORT audit, never a causal correction."""
    features={}
    with Path(feature_path).open(encoding="utf8",newline="") as stream:
        for row in DictReader(stream):
            tag=row.get("tag","").strip()
            if tag and tag.lower() not in ("nan","none"):
                features[row["video_id"]]=tag
    summaries=[]
    for path in (early_path,late_path):
        user_topics=defaultdict(Counter)
        clicks=Counter();observed=Counter()
        rows=0;tagged=0
        with Path(path).open(encoding="utf8",newline="") as stream:
            for row in DictReader(stream):
                rows+=1
                tag=features.get(row["video_id"])
                if not tag:continue
                tagged+=1
                user_topics[row["user_id"]][tag]+=1
                clicks[row["user_id"]]+=int(row["is_click"])
                observed[row["user_id"]]+=1
        summaries.append((user_topics,clicks,observed,rows,tagged))
    a,b=summaries
    def entropy(c):
        n=sum(c.values())
        return -sum((v/n)*log(v/n) for v in c.values() if v) if n else 0
    eligible=[u for u in a[0] if u in b[0] and sum(a[0][u].values())>=5
              and sum(b[0][u].values())>=5]
    # Raw entropy rises with sample count. Balance each user's observations
    # across periods before inspecting any trend. Still NON-CAUSAL.
    def balanced_topic_entropy(user, topics, count, period):
        source=list(topics.elements())
        return entropy(Counter(Random(f"tags:{user}:{period}:20261020").sample(
            source,count)))
    matched=[]
    for user in eligible:
        n=min(sum(a[0][user].values()),sum(b[0][user].values()),30)
        matched.append((balanced_topic_entropy(user,a[0][user],n,"early"),
                        balanced_topic_entropy(user,b[0][user],n,"late")))
    before=mean(x[0] for x in matched) if matched else None
    after=mean(x[1] for x in matched) if matched else None
    raw_before=mean(entropy(a[0][u]) for u in eligible) if eligible else None
    raw_after=mean(entropy(b[0][u]) for u in eligible) if eligible else None
    return {
        "bias_id":"filter_bubble",
        "status":"ABSTAIN_NO_MATCHED_INTERVENTION_TRAJECTORY",
        "source_periods":["standard_2022-04-08_to_04-21",
                          "standard_2022-04-22_to_05-08"],
        "logs":[{"observed_rows":x[3],"with_known_video_tag":x[4]}
                for x in summaries],
        "users_with_at_least_5_tagged_exposures_each_period":len(eligible),
        "mean_tag_entropy_first_period":before,
        "mean_tag_entropy_second_period":after,
        "observed_entropy_change":after-before if before is not None else None,
        "matched_sample_size_cap_per_user_per_period":30,
        "matched_same_number_of_exposures_per_user":True,
        "raw_unbalanced_entropy_change_confounded_by_log_volume":(
            raw_after-raw_before if raw_before is not None else None),
        "independent_confirmation":False,
        "identified_filter_bubble_effect":False,
        "required_next":"Same initial users, timestamped intervention trajectories, "
                        "matched feedback policy with relevance-preserving comparison",
        "note":"Different observational time windows and changing catalogs/user intent "
               "cannot establish filter bubble amplification or Debias causality.",
    }


def run(output):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("Do not replace development evidence")
    bias_registry.verify()
    ds=load_coat(DATA,split_seed=SEED)
    exposure=exposure_real(
        KUAIRAND/"log_random_4_22_to_5_08_pure.csv",
        KUAIRAND/"log_standard_4_22_to_5_08_pure.csv")
    controlled=exposure_synthetic.evaluate(reps=32,events=3000)
    exposure["synthetic_known_propensity_control"]={
        "oracle_vs_estimators":{
            t:{"naive_rmse":v["naive_logged_click_rate"]["rmse"],
               "ips_rmse":v["IPS"]["rmse"],"snips_rmse":v["SNIPS"]["rmse"]}
            for t,v in controlled["targets"].items()},
        "missing_propensity_abstains":
            controlled["missing_propensity_control"]["status"],
        "not_real_world_validation":True}
    selection_path=ROOT/("runs/coat-fivebias-selection-debias-20261020-01/"
                          "coat_selection_development.json")
    selection_source=json.loads(selection_path.read_text(encoding="utf8"))
    if (selection_source["independent_confirmation"] is not False or
        selection_source["test_evaluation_done"] is not False):
        raise ValueError("Selection source unexpectedly used reserved labels")
    comparison={
        policy:{
            "brier":record["development_metrics"]["Brier_random_development"],
            "ndcg_at_5":record["development_metrics"]["NDCG5_random_development"],
            "hit_at_5":record["development_metrics"]["Hit5_random_development"]
        } for policy,record in selection_source["policies"].items()}
    data={
        "selection":{
            "bias_id":"selection",
            "status":"REAL_COAT_PROPENSITY_WEIGHTED_DEVELOPMENT",
            "source":str(selection_path.relative_to(ROOT)),
            "source_sha256":sha256(selection_path.read_bytes()).hexdigest(),
            "selected_development_policy":
                selection_source["selected_development_policy"],
            "policies":comparison,
            "independent_confirmation":False},
        "exposure":exposure,
        "popularity":popularity_coat(ds),
        "filter_bubble":filter_bubble_kuairand(
            KUAIRAND/"video_features_basic_pure.csv",
            KUAIRAND/"log_standard_4_08_to_4_21_pure.csv",
            KUAIRAND/"log_standard_4_22_to_5_08_pure.csv"),
        "fairness":fairness_coat(ds),
    }
    if set(data)!={t.id for t in bias_registry.TASKS}:
        raise ValueError("five-bias registry coverage mismatch")
    output.mkdir(parents=True)
    result={
        "version":VERSION,
        "mode":"fixed_development_protocol_never_confirmation",
        "data":data,
        "confirmed_debias_findings":0,
        "methods_passing_independent_confirmation":0,
        "no_holdout_ratings_scored":True,
        "did_not_relabel_missing_feedback_as_dislike":True,
        "no_qwen_calls":True,
    }
    atomic_json(output/"five_bias_development.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    result=run(args.output)
    print(json.dumps({"status":"FIVE_BIAS_DEVELOPMENT_AUDITED",
                      "statuses":{k:v["status"] for k,v in result["data"].items()},
                      "confirmed_debias_findings":0},ensure_ascii=False))


if __name__=="__main__":
    main()
