"""CPU-only descriptive subgroup audit on the EXISTING ML-1M fixed-candidate setting.
Development targets only; no held-out test, no LLM, no claims of causal fairness.
"""
import argparse, json, math, random, hashlib
from collections import Counter, defaultdict
from pathlib import Path
from rsi_researcher.llmrank.dataset import load_dataset, cases_for, user_utility, _catalog_groups

def js_divergence(p,q):
    m=[(a+b)/2 for a,b in zip(p,q)]
    def kl(a,b):
        return sum(x*math.log(x/y,2) for x,y in zip(a,b) if x>0 and y>0)
    return (kl(p,m)+kl(q,m))/2

def distribution(items, groups, n=5):
    c=Counter(groups.get(i) for i in items)
    s=sum(c.get(g,0) for g in range(n))
    return [c.get(g,0)/s for g in range(n)] if s else [1/n]*n

def mean(xs):
    return sum(xs)/len(xs) if xs else None

def metric(rows):
    result={}
    for subset in ('all','F','M'):
        arr=[r for r in rows if subset=='all' or r['gender']==subset]
        result[subset]=dict(users=len(arr), ndcg5=mean([r['ndcg5'] for r in arr]),
            hr5=mean([r['hit5'] for r in arr]),
            arp5=mean([r['arp5'] for r in arr]),
            head_share5=mean([r['head5'] for r in arr]),
            popularity_calibration_js=mean([r['calib_js'] for r in arr]))
    result['gap_ndcg5_abs']=abs(result['F']['ndcg5']-result['M']['ndcg5'])
    result['worst_group_ndcg5']=min(result['F']['ndcg5'],result['M']['ndcg5'])
    return result

def run(dest,n_each=450,seed=20261022):
    root=Path('/data/cjh/rsi/project')
    users_path=root/'data/external/movielens-1m/users.dat'
    gender={}
    for line in users_path.read_text().splitlines():
        u,g,*_=line.split('::')
        gender[u]=g
    train, positive, known, titles, pop=load_dataset(root/'datasets/llmrank','ml-1m')
    groups=_catalog_groups(pop,titles)[1]
    pool={g:sorted([u for u in train if gender.get(u)==g]) for g in ('F','M')}
    rng=random.Random(seed)
    for g in pool:rng.shuffle(pool[g])
    assert all(len(pool[g])>=n_each for g in ('F','M'))
    selected=pool['F'][:n_each]+pool['M'][:n_each]
    selected.sort()
    cases=cases_for('validation',selected,train,positive,known,titles,seed,n_candidates=20,retrieval='random')
    mx=max(pop.values())
    features={}
    for c in cases:
        u=c['user']
        hist=train[u]
        h_dist=distribution(hist,groups)
        avg_log=mean([math.log1p(pop.get(i,0))/math.log1p(mx) for i in hist])
        features[u]=(h_dist,avg_log)
    records={str(l):[] for l in (0,0.5,1,2,4)}
    for c in cases:
        u=c['user'];hist,mu=features[u]
        for lam in (0,0.5,1,2,4):
            key=str(lam);cal={}
            for i in c['candidates']:
                p=pop.get(i,0)
                norm=math.log1p(p)/math.log1p(mx)
                # A controlled, label-blind score: popularity baseline + optional user-popularity calibration.
                cal[i]=norm-lam*abs(norm-mu)
            rank=sorted(c['candidates'],key=lambda i:(-cal[i],i))
            hit,ndcg=user_utility(rank,c['target'],5)
            top=rank[:5]
            records[key].append(dict(user=u,gender=gender[u],hit5=hit,ndcg5=ndcg,
                arp5=mean([pop.get(i,0) for i in top]),
                head5=mean([float(groups.get(i)==4) for i in top]),
                calib_js=js_divergence(hist,distribution(top,groups))))
    summary={k:metric(rows) for k,rows in records.items()}
    # Stratified user bootstrap: uncertainty in group gap difference and utility difference.
    def bootstrap(arm,draws=600):
        ra=records['0'];rb=records[arm]
        by={g:[(a,b) for a,b in zip(ra,rb) if a['gender']==g] for g in ('F','M')}
        rnd=random.Random(seed+99)
        effects=[];quality=[]
        for _ in range(draws):
            pairs=[]
            for g in ('F','M'):
                bag=by[g]
                pairs.extend(bag[rnd.randrange(len(bag))] for _ in bag)
            beforeF=mean([a['ndcg5'] for a,b in pairs if a['gender']=='F'])
            beforeM=mean([a['ndcg5'] for a,b in pairs if a['gender']=='M'])
            afterF=mean([b['ndcg5'] for a,b in pairs if b['gender']=='F'])
            afterM=mean([b['ndcg5'] for a,b in pairs if b['gender']=='M'])
            effects.append(abs(beforeF-beforeM)-abs(afterF-afterM))
            quality.append(mean([b['ndcg5']-a['ndcg5'] for a,b in pairs]))
        def ci(v):
            v.sort()
            return [v[int(.025*len(v))],v[int(.975*len(v))]]
        return dict(gap_reduction_ci95=ci(effects),ndcg_delta_ci95=ci(quality),
                    bootstrap_draws=draws)
    uncertainty={k:bootstrap(k) for k in records if k!='0'}
    dest=Path(dest)
    dest.mkdir(parents=True,exist_ok=False)
    obj=dict(status='DEVELOPMENT_DESCRIPTIVE_ONLY',dataset='MovieLens-1M',
      provenance='GroupLens original users.dat + already frozen ML-1M interactions',
      seed=seed,balanced_users=n_each*2,per_gender=n_each,
      protocol='random 20 candidate panels with injected held-out positive; validation target only; global train popularity prior and user calibration',
      primary_research_question='observed binary gender-group utility disparity, not causally identified social fairness',
      labels='self-reported gender F/M; no claim to represent all identities',
      ranker='global log(popularity) normalized minus lambda*distance to users train-history mean popularity',
      columns='ndcg5/hr5 one positive per user; full-catalog retrieval NOT measured',
      arms=summary,paired_stratified_user_bootstrap=uncertainty,
      caveat='gender group imbalance is controlled in this panel; candidate sampling, historical exposure, and self-reported labels limit fairness interpretation',
      qwen_calls=0,holdout_opened=False)
    (dest/'results.json').write_text(json.dumps(obj,indent=2))
    print('ML_GROUP_FAIRNESS',json.dumps({k:{'overall_ndcg':round(v['all']['ndcg5'],4),
         'gap_ndcg':round(v['gap_ndcg5_abs'],4),'worst_group_ndcg':round(v['worst_group_ndcg5'],4),
         'popcalib_js':round(v['all']['popularity_calibration_js'],4)}
         for k,v in summary.items()}))
    print('DIR',dest)

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--output',required=True)
    a.add_argument('--users-per-group',type=int,default=450)
    a.add_argument('--seed',type=int,default=20261022)
    v=a.parse_args()
    run(v.output,v.users_per_group,v.seed)
