"""Near-fully-observed KuaiRec LLM Qwen position-sensitivity experiment.
Frozen, balanced positive/negative sampled evaluation slates; time-separated user history.
Evaluation probes only, not a learned/confirmed debias discovery.
"""
import argparse,csv,hashlib,io,json,random,statistics,zipfile,math
from collections import Counter,defaultdict
from pathlib import Path
from rsi_researcher.llmrank.agents import RankGateway
from rsi_researcher.llmrank.strategy import rank,BASELINE
from rsi_researcher.runtime.models import create_model
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.client import atomic_json,usage_summary

ARCHIVE=Path('/data/cjh/rsi/project/data/external/kuairec/KuaiRec.zip')

def fixed_hash(seed,u):
    return int.from_bytes(hashlib.sha256(f'{seed}:{u}'.encode()).digest()[:8],'big')

def dcg(pred,truth,k=5):
    ideal=sum(1/math.log2(i+2) for i in range(min(k,sum(truth.values()))))
    if pred is None or ideal==0:return 0.
    return sum(truth.get(item,0)/math.log2(i+2) for i,item in enumerate(pred[:k]))/ideal

def experiment(output,n,seed):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as z:
        with z.open('KuaiRec 2.0/data/kuairec_caption_category.csv') as f:
            rd=csv.DictReader(io.TextIOWrapper(f,encoding='utf-8'))
            titles={}
            for r in rd:
                text=' '.join(x for x in (r['second_level_category_name'],r['caption']) if x and x!='UNKNOWN').strip()
                titles[r['video_id']]=text[:95] if text else 'Video ID '+r['video_id']
        ids=set()
        with z.open('KuaiRec 2.0/data/small_matrix.csv') as f:
            for row in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8')):
                ids.add(int(row['user_id']))
        # Not any of the already evaluated selection split's test users.
        candidates=[u for u in sorted(ids) if fixed_hash(20261022,u)%5!=0 and fixed_hash(20261023,u)%5!=0]
        candidates=sorted(candidates,key=lambda u:fixed_hash(seed,u))[:n+18]
        observed=defaultdict(list)
        with z.open('KuaiRec 2.0/data/small_matrix.csv') as f:
            for row in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8')):
                uid=int(row['user_id'])
                if uid in candidates and row['timestamp']:
                    observed[uid].append((float(row['timestamp']),row['video_id'],int(float(row['watch_ratio'])>=1.0)))
    rng=random.Random(seed)
    cases=[]
    labels={}
    pop=Counter()
    for u in candidates:
        seq=sorted(observed[u])
        if len(seq)<100: continue
        split=int(.8*len(seq))
        history=[i for t,i,y in seq[:split] if y==1]
        future_pos=[i for t,i,y in seq[split:] if y==1]
        future_neg=[i for t,i,y in seq[split:] if y==0]
        # Restrict evaluation to one unique label per video to avoid duplicate contradictions.
        positive=set(future_pos)
        negative=set(future_neg)-positive
        future_pos=[i for i in sorted(positive) if i not in set(history)]
        future_neg=sorted(negative)
        if len(future_pos)<10 or len(future_neg)<10 or not history:continue
        sampled_pos=rng.sample(future_pos,10)
        sampled_neg=rng.sample(future_neg,10)
        slate=sampled_pos+sampled_neg
        rng.shuffle(slate)
        history=history[-50:]
        pop.update(history)
        case={'user':str(u),'history':history,'candidates':slate,'target':sampled_pos[0]}
        cases.append(case)
        labels[str(u)]={i:1 for i in sampled_pos}
        if len(cases)>=n:break
    assert len(cases)==n,(len(cases),n)
    config=json.loads(Path('/data/cjh/rsi/project/configs/llmrank_qwen.json').read_text())
    modeldir=out/'model';modeldir.mkdir()
    limits=RunLimits(modeldir,max_requests=n*3,max_tokens=600000,max_seconds=3600)
    model=create_model(modeldir,limits.max_requests,limits,config)
    gateway=RankGateway(model,cache_path=out/'rank_cache.jsonl')
    rows=[]
    for k,case in enumerate(cases):
        order=list(case['candidates'])
        reverse=order[::-1]
        a,qa=rank(gateway,case,order,titles,BASELINE,pop,seed+k*101,f'KREC.{k}.random',
                  max_requests=n*3,return_details=True)
        b,qb=rank(gateway,case,reverse,titles,BASELINE,pop,seed+k*101,f'KREC.{k}.reverse',
                  max_requests=n*3,return_details=True)
        z=labels[case['user']]
        ma=dcg(a,z)
        mb=dcg(b,z)
        rows.append({'user':case['user'],'random_ndcg5':ma,'reversed_ndcg5':mb,
           'random_p5':(sum(z.get(i,0) for i in a[:5])/5 if a else 0),
           'reversed_p5':(sum(z.get(i,0) for i in b[:5])/5 if b else 0),
           'overlap_top5':(len(set(a[:5])&set(b[:5]))/5 if a and b else None),
           'valid_pair':a is not None and b is not None,
           'repair_random':qa['repaired_votes'],'repair_reverse':qb['repaired_votes']})
        if k%10==9:print('KUAIREC_POSITION_PROGRESS',k+1,'of',n,'calls',gateway.calls,flush=True)
    valid=[x for x in rows if x['valid_pair']]
    assert len(valid)>n//2
    diffs=[x['reversed_ndcg5']-x['random_ndcg5'] for x in valid]
    rnd=random.Random(seed+77)
    draws=500
    boot=sorted(sum(diffs[rnd.randrange(len(diffs))] for _ in diffs)/len(diffs) for j in range(draws))
    result={
     'status':'REAL_QWEN_POSITION_ORDER_SENSITIVITY_ON_KUAIREC',
     'data_source':'KuaiRec 2.0 nearly complete ratings + caption/category side information',
     'seed':seed,'users':n,'paired_valid_users':len(valid),
     'development_only':True,
     'missing_timestamp_handling':'exclude records without timestamps from chronological probe; KuaiRec small matrix about 3.9% timestamp missing',
     'split':'within user first 80% timestamp historical engagements, last 20% future candidate labels',
     'candidates':'20 future items, 10 positive watch_ratio>=1 and 10 negative, evaluation-only balanced panel. Models do not see class labels.',
     'position_arms':'frozen shuffled slate versus exact reversed slate, same item IDs and 50 historical positive items',
     'score':'all 10 known positive items from fully-observed panel, NDCG@5 normalized by ideal five hits',
     'NDCG5_random':statistics.mean(r['random_ndcg5'] for r in valid),
     'NDCG5_reversed':statistics.mean(r['reversed_ndcg5'] for r in valid),
     'P5_random':statistics.mean(r['random_p5'] for r in valid),
     'P5_reversed':statistics.mean(r['reversed_p5'] for r in valid),
     'Top5_set_overlap':statistics.mean(r['overlap_top5'] for r in valid),
     'delta_ndcg5_ci95':[boot[int(.025*draws)],boot[int(.975*draws)]],
     'repaired_arms':sum(x['repair_random']+x['repair_reverse']>0 for x in rows),
     'qwen_requests':gateway.calls,'qwen_cache_hits':gateway.cache_hits,
     'rows':rows,
     'evidence_scope':'Within-slate model input-order sensitivity; not logging exposure or actual online display slot bias',
     'confirmed_bias_reduction':False}
    atomic_json(out/'results.json',result)
    atomic_json(out/'usage.json',usage_summary(modeldir))
    print('KUAIREC_POSITION_RESULT',json.dumps({k:result[k] for k in
        ('users','paired_valid_users','NDCG5_random','NDCG5_reversed','Top5_set_overlap',
         'delta_ndcg5_ci95','qwen_requests','repaired_arms')}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--output',required=True);p.add_argument('--users',type=int,default=40)
 p.add_argument('--seed',type=int,default=20261024)
 a=p.parse_args();experiment(a.output,a.users,a.seed)
