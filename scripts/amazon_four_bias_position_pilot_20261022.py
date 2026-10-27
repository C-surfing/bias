"""Frozen-label Amazon Qwen3 position probe + popularity diagnostics.
Dev only, no held-out testing or RSI winner selection. Isolated output directory.
"""
import argparse, json, random, math, traceback
from collections import Counter
from pathlib import Path
from rsi_researcher.llmrank.dataset import load_dataset,cases_for,summarize,cpb
from rsi_researcher.llmrank.strategy import BASELINE,evaluate_strategy
from rsi_researcher.llmrank.agents import RankGateway
from rsi_researcher.runtime.models import create_model
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.client import atomic_json,usage_summary

def boot_pairs(rows,first,draws=400,seed=27):
    rng=random.Random(seed)
    n=len(rows)
    d=[]
    for _ in range(draws):
        ix=[rng.randrange(n) for _ in range(n)]
        def nd(r):
            return sum((0 if r[i]['ranking'] is None or r[i]['target'] not in r[i]['ranking'][:5]
                 else 1/math.log2(r[i]['ranking'].index(r[i]['target'])+2)) for i in ix)/n
        d.append(nd(first)-nd(rows))
    d.sort()
    return {'first_minus_random_ndcg5_ci95':[d[int(draws*.025)],d[int(draws*.975)]],
            'paired_users':n,'resamples':draws}

def rerank(rows,pop,lam):
    out=[]
    for row in rows:
        rank=row['ranking']
        if rank is None:
            out.append(row.copy())
            continue
        mx=max(pop.get(i,0) for i in rank)
        scale=math.log1p(mx) or 1.
        n=len(rank)
        rank2=sorted(rank,key=lambda i:( -(1-rank.index(i)/n-lam*math.log1p(pop.get(i,0))/scale),i))
        out.append({**row,'ranking':rank2})
    return out

def one(dataset,root,seed,n,config):
    d=root/dataset
    d.mkdir(parents=True,exist_ok=False)
    path=Path('/data/cjh/rsi/project/datasets/amazon/processed')/dataset
    train,positive,known,titles,pop=load_dataset(path,dataset)
    users=sorted(train)
    random.Random(seed).shuffle(users)
    # Need to ensure no user from this study appears in the old 20261021 Beauty cohort.
    # Exactly this NEW random validation panel; no holdout labels loaded by cases_for.
    cohort=users[:n]
    cases=cases_for('validation',cohort,train,positive,known,titles,seed,20,'bm25')
    modeldir=d/'qwen_model'
    modeldir.mkdir()
    limits=RunLimits(modeldir,max_requests=n*3,max_tokens=900000,max_seconds=3600)
    model=create_model(modeldir,limits.max_requests,limits,config)
    gw=RankGateway(model,cache_path=d/'rank_cache.jsonl')
    raw=evaluate_strategy(gw,cases,titles,pop,BASELINE,seed,'amazon_position_dev',max_requests=n*3)
    sm=raw['metrics']['5']
    abst=abs(sm['CandidatePositionBias_NDCG@5']) if sm['CandidatePositionBias_NDCG@5'] is not None else None
    first=summarize(raw['first'],pop,titles,5)
    overlap=sum((len(set(a['ranking'][:5])&set(b['ranking'][:5]))/5)
         for a,b in zip(raw['random'],raw['first'])
         if a['ranking'] is not None and b['ranking'] is not None)
    pair_valid=sum(a['ranking'] is not None and b['ranking'] is not None
           for a,b in zip(raw['random'],raw['first']))
    penalties={}
    for lam in (0.5,1.,2.):
        revised=rerank(raw['random'],pop,lam)
        penalties[str(lam)]=summarize(revised,pop,titles,5)
    report={
      'status':'DEVELOPMENT_DESCRIPTIVE_NO_HOLDOUT','model':'qwen3-8b_vllm_18024',
      'dataset':dataset,'seed':seed,'n_users':n,
      'relevance':'validation leave-two-out target is inserted once into frozen 20 TitleBM25 candidates',
      'baseline_random':sm,'target_first':first,
      'position_bias_abs_ndcg5':abst,
      'position_first_minus_random':boot_pairs(raw['random'],raw['first'],seed=seed),
      'top5_set_overlap_first_random':overlap/pair_valid if pair_valid else None,
      'paired_valid_users':pair_valid,'qwen_requests':model.requests,
      'qwen_cache_hits':gw.cache_hits,
      'popularity_rerank_offline':penalties,
      'support':'PopREO group support 5 is required for development gate; if any group below 5 this pilot is inconclusive for popularity',
      'no_heldout_access':True,
      'evidence_grade':'pilot exploratory; sampled users can overlap earlier development but never previous holdout is consulted for experiment decisions'
    }
    atomic_json(d/'results.json',report)
    atomic_json(d/'usage_summary.json',usage_summary(modeldir))
    print('RESULT',dataset,json.dumps({
      'n':n,'qwen_requests':model.requests,'valid_rankings':sm['valid_rankings'],
      'ndcg5':sm['NDCG@5'],'hr5':sm['HR@5'],
      'popreo5':sm['PopREO@5'],'positive_counts':sm['popreo_group_positive_counts'],
      'cpb_ndcg5':sm['CandidatePositionBias_NDCG@5'],
      'first_ndcg5':first['NDCG@5'],
      'top5_overlap':report['top5_set_overlap_first_random'],
      'offline_penalty_1_arp':penalties['1.0']['ARP@5'],
      'offline_penalty_1_popreo':penalties['1.0']['PopREO@5']}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--users',type=int,default=56)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--datasets',nargs='+',default=['Beauty','Sports','CDs'])
    a=p.parse_args()
    root=Path(a.output)
    root.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('/data/cjh/rsi/project/configs/llmrank_qwen.json').read_text())
    for i,dataset in enumerate(a.datasets):
        try: one(dataset,root,a.seed+i,a.users,config)
        except Exception as ex:
            (root/f'{dataset}.ERROR.txt').write_text(traceback.format_exc())
            print('FAILED',dataset,str(ex),flush=True)
    print('COMPLETE',root,flush=True)

if __name__=='__main__':main()
