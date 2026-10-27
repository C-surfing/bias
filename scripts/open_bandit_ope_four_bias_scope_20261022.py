"""Open Bandit official 10k random/BTS (campaign=all).
Out-of-policy evaluation of predeclared static per-position policies with known random log propensity.
No Qwen, no social fairness, no unsupported full-catalog NDCG.
"""
import argparse,csv,json,math,random,statistics
from collections import Counter,defaultdict
from pathlib import Path
DATA=Path('/data/cjh/rsi/project/data/external/open-bandit-demo')
def read(arm):
    with (DATA/(arm+'_all.csv')).open() as f:
        return [dict(item=int(r['item_id']),pos=int(r['position']),click=int(r['click']),
                     p=float(r['propensity_score'])) for r in csv.DictReader(f)]
def bounded(vals):
    vals=sorted(vals)
    return [vals[int(.025*len(vals))],vals[min(len(vals)-1,int(.975*len(vals)))]] if vals else None
def summarize(selected,bts,pi,alpha=25):
    # Independent reward model fitted only to BTS data, distinct from heldout random logger
    stats=defaultdict(lambda:[0,0])
    pooled=[0,0]
    for r in bts:
        q=stats[(r['pos'],r['item'])];q[0]+=r['click'];q[1]+=1
        pooled[0]+=r['click'];pooled[1]+=1
    base=pooled[0]/pooled[1]
    pred={(pos,item):(stats[(pos,item)][0]+alpha*base)/(stats[(pos,item)][1]+alpha)
          for pos in (1,2,3) for item in range(80)}
    dm={pos:sum(pi[pos][j]*pred[(pos,j)] for j in range(80)) for pos in (1,2,3)}
    contrib=[]
    for r in selected:
        action=r['item'];p=r['pos'];q=pred[p,action];w=pi[p][action]/r['p']
        contrib.append((w,w*r['click'],dm[p]+w*(r['click']-q),dm[p],r['click'],p))
    def metrics(chunk):
        ws=sum(x[0] for x in chunk)
        return dict(
          IPS=sum(x[1] for x in chunk)/len(chunk),
          SNIPS=sum(x[1] for x in chunk)/ws if ws else None,
          DR=sum(x[2] for x in chunk)/len(chunk),
          DM=sum(x[3] for x in chunk)/len(chunk),
          ESS=ws**2/sum(x[0]**2 for x in chunk) if ws else 0,
          weight_mean=ws/len(chunk),
          empirical_reward=sum(x[4] for x in chunk)/len(chunk),
          overlap_nonzero=sum(x[0]>0 for x in chunk)/len(chunk))
    result=metrics(contrib)
    rng=random.Random(20261022)
    bs={a:[] for a in ('IPS','SNIPS','DR')}
    for _ in range(500):
        s=[contrib[rng.randrange(len(contrib))] for k in range(len(contrib))]
        tmp=metrics(s)
        for k in bs:
            if tmp[k] is not None: bs[k].append(tmp[k])
    result['ci95']={k:bounded(v) for k,v in bs.items()}
    return result

def run(output):
    rand=read('random')
    bts=read('bts')
    assert len(rand)==len(bts)==10000
    assert all(abs(x['p']-1/80)<1e-8 for x in rand)
    assert all(x['p']>0 for x in bts)
    counts=defaultdict(Counter)
    for row in bts: counts[row['pos']][row['item']]+=1
    policies={}
    for name in ('uniform','bts_head20','bts_tail20','bts_90head10uniform'):
        pa={}
        for p in (1,2,3):
            ranking=sorted(range(80),key=lambda i:(-counts[p][i],i))
            if name=='uniform': vals=[1/80]*80
            elif name=='bts_head20':
                hs=set(ranking[:20]);vals=[int(i in hs)/20 for i in range(80)]
            elif name=='bts_tail20':
                ts=set(ranking[-20:]);vals=[int(i in ts)/20 for i in range(80)]
            else:
                hs=set(ranking[:20]);vals=[.1/80+.9*int(i in hs)/20 for i in range(80)]
            assert abs(sum(vals)-1)<1e-8
            pa[p]=vals
        policies[name]=pa
    result={name:summarize(rand,bts,policy) for name,policy in policies.items()}
    # Display slot 1/2/3 is not LLM candidate ordering; item randomization ≠ random assignment to slot.
    slot={}
    for arm,rows in (('random',rand),('bts',bts)):
        slot[arm]={}
        for p in (1,2,3):
            rr=[r for r in rows if r['pos']==p]
            slot[arm][str(p)]={'n':len(rr),'clicks':sum(r['click'] for r in rr),
                    'click_rate':statistics.mean(r['click'] for r in rr)}
    path=Path(output);path.mkdir(parents=True,exist_ok=False)
    obj={'status':'REAL_KNOWN_PROPENSITY_OPE_DEVELOPMENT_EXPLORATORY',
         'dataset':'Open Bandit 10k (campaign=all, random/BTS)',
         'logging_reference':'random behavior, pscore=0.0125 for every item, 80 actions',
         'target_policies':'static per-slot distributions defined exclusively from separate BTS item exposure counts (not actions or rewards in random logger)',
         'reward':'click binary (rare)', 'estimators':['IPS','SNIPS','DM','DR'],
         'position_1_3':'physical display slots; not equivalent to LLM candidate prompt-order position',
         'assumption':'action randomized within observed display slot and no policy-dependent unmeasured reward changes; no causal display-slot comparison',
         'n_random':len(rand),'n_bts':len(bts),'results':result,
         'position_diagnostics':slot,
         'same_environment_caution':'Random vs BTS logs may represent distinct user distributions and sequential online contexts; confirmability limited',
         'discoveries_confirmed':0,'qwen_calls':0,'holdout_untouched':True}
    (path/'results.json').write_text(json.dumps(obj,indent=2))
    print('OPEN_BANDIT_RESULTS',json.dumps({k:{key:round(v[key],6) for key in ('IPS','SNIPS','DR','ESS','overlap_nonzero','weight_mean')} for k,v in result.items()}),flush=True)
    print('OPEN_BANDIT_POSITION',json.dumps(slot),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
