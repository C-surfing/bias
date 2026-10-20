"""Paired sequential recommendation experiment with known synthetic user preferences.

The SAME initial user states and potential-outcome uniforms are used for
baseline and diversity policies, as well as for a no-feedback placebo arm.
This identifies policy effects WITHIN the simulator, not real-world bubbles.
"""
from math import exp,log
from random import Random
from statistics import mean
from pathlib import Path
import argparse,json

from rsi_researcher.runtime.client import atomic_json

TOPICS=6
ROUNDS=50
USERS_PER_SEED=60
SEEDS=32
EXPLORATION=.22
SEED=20261020


def sample(probs, u):
    accum=0.0
    for j,p in enumerate(probs):
        accum+=p
        if u<accum:
            return j
    return len(probs)-1


def entropy(counts):
    n=sum(counts)
    return -sum((v/n)*log(v/n) for v in counts if v)/log(TOPICS) if n else 0.


def probabilities(memory, *, explore=0.):
    # Algorithm sees only *past clicked feedback*, not latent truth.
    scores=[(x+1.)**1.35 for x in memory]
    z=sum(scores)
    return [(1-explore)*score/z+explore/TOPICS for score in scores]


def paired_user(seed,user,explore):
    rng=Random(f"{seed}:{user}")
    # True preferences fixed through both trajectories; warm-start reflects
    # only partial, biased history, and cannot be read by the policy.
    preferred=rng.randrange(TOPICS)
    truth=[.43+.13*(i==preferred)+rng.uniform(-.045,.045)
           for i in range(TOPICS)]
    initial=[1.]*TOPICS
    initial[preferred]=rng.uniform(6,12)
    other=(preferred+1+rng.randrange(TOPICS-1))%TOPICS
    initial[other]=rng.uniform(3,5)
    noise=[(rng.random(),rng.random()) for _ in range(ROUNDS)]
    outcomes=[]
    for alpha,feedback in ((0.,True),(explore,True),(0.,False)):
        memory=list(initial);exposure=[0]*TOPICS;clicks=0
        exp_reward=0.
        for u,v in noise:
            probs=probabilities(memory,explore=alpha)
            topic=sample(probs,u)
            exposure[topic]+=1
            reward=int(v<truth[topic])
            exp_reward+=truth[topic]
            clicks+=reward
            if feedback and reward:
                memory[topic]+=1.
        outcomes.append({
            "exposure_entropy":entropy(exposure),
            "unique_topics":sum(c>0 for c in exposure),
            "mean_expected_utility":exp_reward/ROUNDS,
            "observed_click_rate":clicks/ROUNDS,
        })
    baseline,treatment,placebo=outcomes
    return {
        "entropy_gain":treatment["exposure_entropy"]-baseline["exposure_entropy"],
        "utility_gain":treatment["mean_expected_utility"]-baseline["mean_expected_utility"],
        "click_gain":treatment["observed_click_rate"]-baseline["observed_click_rate"],
        "feedback_amplification":placebo["exposure_entropy"]-baseline["exposure_entropy"],
        "baseline":baseline,"treated":treatment,"no_feedback":placebo
    }


def paired_ci(values,seed):
    rng=Random(seed)
    n=len(values)
    # Paired bootstrap resamples users/cohorts, not individual time steps.
    vals=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n
                for _ in range(500))
    return [vals[12], vals[487]]


def evaluate(*, seeds=SEEDS,users=USERS_PER_SEED,explore=EXPLORATION):
    if seeds<4 or users<4 or not 0<explore<.7:
        raise ValueError("invalid preregistered controlled feedback simulator")
    # Fixed disjoint *new seeds* for confirmation; no selecting lucky seeds.
    panels={"development":range(SEED,SEED+seeds),
            "unseen_simulation":range(SEED+10000,SEED+10000+seeds)}
    output={}
    for name,seed_values in panels.items():
        user_rows=[paired_user(seed,user,explore)
                   for seed in seed_values for user in range(users)]
        report={}
        for key in ("entropy_gain","utility_gain","click_gain",
                    "feedback_amplification"):
            xs=[r[key] for r in user_rows]
            report[key]={"mean":mean(xs),
                         "paired_user_bootstrap_ci":paired_ci(xs,SEED+len(key))}
        report["users"]=len(user_rows)
        report["per_user_time_steps"]=ROUNDS
        output[name]=report
    # The alpha, utility bound and novelty metric are frozen for BOTH splits,
    # NOT chosen after seeing the blind simulation panel.
    d=output["development"];h=output["unseen_simulation"]
    tests={
        "development_entropy_gain_min_0_08":d["entropy_gain"]["mean"]>=.08,
        "unseen_entropy_gain_min_0_08":h["entropy_gain"]["mean"]>=.08,
        "unseen_entropy_bootstrap_lower_above_zero":
             h["entropy_gain"]["paired_user_bootstrap_ci"][0]>0,
        "development_utility_loss_max_0_025":
             d["utility_gain"]["mean"]>=-.025,
        "unseen_utility_loss_max_0_025":
             h["utility_gain"]["mean"]>=-.025,
        "baseline_feedback_trend_stronger_than_no_feedback":
             h["feedback_amplification"]["mean"]>=0,
    }
    return {
        "bias_id":"filter_bubble",
        "status":"SYNTHETIC_PAIRED_SEQUENTIAL_INTERVENTION",
        "user_potential_outcomes_known_to_evaluator_only":True,
        "recommender_inputs":"initial history and previously observed clicks",
        "same_initial_state":True,
        "same_exogenous_uniforms_across_policies":True,
        "negative_control":"frozen-history recommendation without feedback updates",
        "intervention":"mix recommendation-topic sampling with fixed uniform exploration",
        "explore_alpha":explore,
        "fixed_acceptance":tests,
        "simulator_evidence_passes":all(tests.values()),
        "panels":output,
        "no_real_world_causal_claim":True,
        "independent_real_data_confirmation":False,
    }


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    result=evaluate()
    p=Path(args.output)
    p.mkdir(parents=True,exist_ok=False)
    atomic_json(p/"paired_feedback.json",result)
    print(json.dumps({"simulator_passes":result["simulator_evidence_passes"],
                      "fixed_checks":result["fixed_acceptance"],
                      "unseen":result["panels"]["unseen_simulation"]},
                     ensure_ascii=False))


if __name__=="__main__":
    main()
