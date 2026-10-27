"""Mechanism test of longitudinal filter bubbles with REAL KuaiRand topic priors
and explicitly SIMULATED feedback trajectories. No claim of real-world causality."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,math,json,random,statistics
SOURCE=Path('/data/cjh/rsi/project/data/external/kuairand-pure/KuaiRand-Pure/data')

def entropy(p):
    return -sum(x*math.log(x) for x in p if x>0) / math.log(len(p))

def sample(weights,u):
    cum=0.
    for i,w in enumerate(weights):
        cum+=w
        if cum>=u:return i
    return len(weights)-1

def metric(seed,eta,users=240,steps=40):
    labels={}
    freq=Counter()
    with (SOURCE/'video_features_basic_pure.csv').open() as f:
        for row in csv.DictReader(f):
            topic=row['tag'].split(',')[0].strip()
            labels[row['video_id']]=topic
            freq[topic]+=1
    topics=[t for t,n in freq.most_common(12)]
    index={t:i for i,t in enumerate(topics)}
    positives=defaultdict(Counter)
    with (SOURCE/'log_standard_4_08_to_4_21_pure.csv').open() as f:
        for row in csv.DictReader(f):
            if row['is_click']=='1':
                topic=labels.get(row['video_id'])
                if topic in index: positives[row['user_id']][topic]+=1
    users_list=sorted([u for u,c in positives.items() if sum(c.values())>=15])
    rng=random.Random(seed)
    rng.shuffle(users_list)
    users_list=users_list[:users]
    assert len(users_list)==users
    results={}
    for policy in ('greedy','five_topics'):
        rows=[]
        for uid in users_list:
            c=positives[uid]
            initial=[(c[t]+1)/(sum(c[z] for z in topics)+len(topics)) for t in topics]
            theta=initial[:]
            estimated=initial[:]
            exposure=Counter()
            rewards=0
            peruser=random.Random(f'{seed}:{uid}').random
            for t in range(steps):
                order=sorted(range(len(topics)),key=lambda i:(-estimated[i],i))
                if policy=='greedy':
                    choices=[order[0]]*5
                else:
                    choices=order[:5]
                for i in choices:exposure[i]+=1
                # Draw choice from current user interest; unlike a real log,
                # this is a controlled generative assumption.
                weights=[theta[i] for i in choices]
                z=sum(weights)
                picked=choices[sample([w/z for w in weights],peruser())]
                # Independent stochastic engagement, with common draws per user & policy.
                reward=1 if peruser() < min(.95,.1+1.8*theta[picked]) else 0
                rewards+=reward
                if reward:
                    estimated=[.95*x for x in estimated]
                    estimated[picked]+=.05
                    estimated=[x/sum(estimated) for x in estimated]
                    theta=[(1-eta)*x for x in theta]
                    theta[picked]+=eta
            rows.append({'user':uid,'reward_per_step':rewards/steps,
                'final_interest_entropy':entropy(theta),
                'initial_interest_entropy':entropy(initial),
                'entropy_change':entropy(theta)-entropy(initial),
                'exposure_entropy':entropy([exposure[i]/(steps*5) for i in range(len(topics))]),
                'exposure_topic_coverage':sum(exposure[i]>0 for i in range(len(topics)))/len(topics)})
        results[policy]=dict(reward=statistics.mean(r['reward_per_step'] for r in rows),
            interest_entropy_delta=statistics.mean(r['entropy_change'] for r in rows),
            exposure_entropy=statistics.mean(r['exposure_entropy'] for r in rows),
            topic_coverage=statistics.mean(r['exposure_topic_coverage'] for r in rows))
    return results

def run(path):
    results={}
    for seed in (20261022,20261023,20261024):
        for eta in (0.0,0.06,0.15):
            results[f'{seed}:{eta}']=metric(seed,eta)
    out=Path(path);out.mkdir(parents=True,exist_ok=False)
    payload={'status':'SEMI_SYNTHETIC_FILTER_BUBBLE_MECHANISM_ONLY',
        'data_source':'KuaiRand real early observed clicks initialize user topic preference, video tag used for topic',
        'model':'40 rounds, 5 topics/items per slate, simulated relevance, click feedback, and interest adaptation',
        'preference_update':'theta=(1-eta)*theta + eta*onehot(clicked_topic) only on simulated engagement',
        'eta_zero_negative_control':'If eta=0, true interest entropy must not contract even if recommendation exposure becomes narrow',
        'policy_arms':['greedy repeated single topic','five distinct top topics'],
        'reality_limitation':'Endogenous adaptation and simulated clicks are researcher assumptions, not observed user counterfactuals',
        'results':results,'real_debias_confirmed':False,'qwen_calls':0}
    (out/'results.json').write_text(json.dumps(payload,indent=2))
    print('FILTER_BUBBLE',json.dumps(results))
    print('DIR',out)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',required=True);v=a.parse_args();run(v.output)
