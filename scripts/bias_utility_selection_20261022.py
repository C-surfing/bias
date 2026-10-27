"""Selection-bias semisynthetic identification benchmark using ML-1M REAL ratings.
The observed MovieLens ratings themselves are MNAR; only our additional mask is known.
Independent users for train/test; no Qwen, no heldout tuning.
"""
import argparse,hashlib,json,math,random,statistics,zipfile
from collections import defaultdict
from pathlib import Path
SOURCE=Path('/data/cjh/rsi/project/data/external/movielens-1m/ml-1m.zip')

def fit(rows,weighted=False,clip=None):
    counts=defaultdict(float);positive=defaultdict(float);global_n=0;global_y=0
    for u,i,y,prop in rows:
        w=1/prop if weighted else 1.
        if clip:w=min(w,clip)
        counts[i]+=w
        positive[i]+=w*y
        global_n+=w
        global_y+=w*y
    base=global_y/global_n
    smoothed={i:(positive[i]+20*base)/(counts[i]+20) for i in counts}
    return smoothed,base

def evalscore(rows,score,prior):
    brier=0;n=0;byuser=defaultdict(list)
    for u,i,y,_ in rows:
        p=score.get(i,prior)
        brier+=(p-y)**2;n+=1
        byuser[u].append((p,i,y))
    ndcg=[]
    for u,rs in byuser.items():
        rank=sorted(rs,key=lambda r:(-r[0],r[1]))[:5]
        pos=sum(y for p,i,y in rs)
        if pos==0:continue
        dcg=sum(y/math.log2(j+2) for j,(p,i,y) in enumerate(rank))
        idcg=sum(1/math.log2(j+2) for j in range(min(5,pos)))
        ndcg.append(dcg/idcg)
    return {'Brier':brier/n,'NDCG5_rated_item_panel':statistics.mean(ndcg),
            'rating_count':n,'evaluated_users':len(ndcg)}

def run(out,seed=20261022):
    with zipfile.ZipFile(SOURCE) as z:
        raw=z.read('ml-1m/ratings.dat').decode()
    allrows=[]
    users=set()
    for ln in raw.splitlines():
        u,i,r,t=ln.split('::')
        allrows.append((u,i,int(int(r)>=4)))
        users.add(u)
    rng=random.Random(seed)
    selected=sorted(users)
    rng.shuffle(selected)
    cut=int(.8*len(selected))
    trainusers=set(selected[:cut])
    train=[];test=[]
    for u,i,y in allrows:
        if u in trainusers:
            propensity=.12+.76*y # known extra synthetic selection, different rates by outcome
            if rng.random() < propensity:
                train.append((u,i,y,propensity))
        else:test.append((u,i,y,1.))
    res={}
    for name,weighted,clip in [('naive',False,None),('known_mask_IPW',True,None),
                                ('IPW_clip_5',True,5.),('IPW_clip_2',True,2.)]:
        score,prior=fit(train,weighted,clip)
        res[name]=evalscore(test,score,prior)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    payload={'status':'SEMI_SYNTHETIC_SELECTION_BIAS_CONTROLLED','dataset':'MovieLens1M original ratings',
       'seed':seed,'train_users':len(trainusers),'test_users':len(users)-len(trainusers),
       'train_rows_extra_masked':len(train),'test_rows_originally_observed':len(test),
       'extra_selection_process':{'p_obs_if_rating_ge4':.88,'p_obs_else':.12},
       'mask_known_by_experimenter':True,
       'mechanism':'Training ratings additionally hidden under known label-dependent propensity, then train identical smoothed item means with/without oracle IPW.',
       'reference_scope':'Original self-selected ratings of untouched users; not full universe of unobserved preferences',
       'results':res,'independent_real_bias_identification':False,
       'interpretation':'Only tests correction of artificially imposed known MNAR mask conditional on an already MNAR ratings corpus',
       'qwen_calls':0,'heldout_post_selection':False}
    (out/'results.json').write_text(json.dumps(payload,indent=2))
    print('SELECTION',json.dumps({k:{z:round(v[z],6) for z in ('Brier','NDCG5_rated_item_panel')} for k,v in res.items()}))
    print('DIR',out)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20261022)
 a=p.parse_args();run(a.output,a.seed)
