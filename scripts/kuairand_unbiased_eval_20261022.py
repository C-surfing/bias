"""KuaiRand-Pure: evaluate two observational estimators on independent user-heldout RANDOM exposures.
Pure random insertion log is a reference sampling frame, not a complete OPE oracle.
"""
import argparse,csv,hashlib,json,math,random,statistics
from collections import Counter,defaultdict
from pathlib import Path
BASE=Path('/data/cjh/rsi/project/data/external/kuairand-pure/KuaiRand-Pure/data')
RFILE=BASE/'log_random_4_22_to_5_08_pure.csv'
SFILE=BASE/'log_standard_4_22_to_5_08_pure.csv'

def split(seed,user):
    x=hashlib.blake2b(f'{seed}:{user}'.encode(),digest_size=4).digest()
    return int.from_bytes(x,'big')%5==0
def rows(path,seed,eval_users):
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['tab']!='1':continue
            if (row['is_rand']=='1') != ('log_random_' in path.name):
                raise ValueError('random flag inconsistent')
            is_val=split(seed,row['user_id'])
            if is_val!=eval_users:continue
            yield int(row['video_id']),int(row['is_click']),row['user_id']
def fitted(rows_):
    clicks=Counter()
    nums=Counter()
    globalclick=0
    for item,y,user in rows_:
        nums[item]+=1
        clicks[item]+=y
        globalclick+=y
    prior=globalclick/max(1,sum(nums.values()))
    score={item:(clicks[item]+20*prior)/(nums[item]+20) for item in nums}
    return score,prior,nums
def result(truth,score,prior,groups):
    b=0.;a=0.;n=0
    for item,y,user in truth:
        val=score.get(item,prior)
        b+=(val-y)**2
        a+=min(.9999,max(.0001,val))
        n+=1
    # Need measured relative calibrations and separate head/tail Brier
    gr=defaultdict(lambda:[0,0,0])
    for item,y,user in truth:
        p=score.get(item,prior)
        k='head' if item in groups else 'nonhead'
        gr[k][0]+=(p-y)**2
        gr[k][1]+=y
        gr[k][2]+=1
    return {'Brier':b/n,'n_eval':n,
        'brier_by_group':{k:v[0]/v[2] for k,v in gr.items()},
        'ctr_by_group':{k:v[1]/v[2] for k,v in gr.items()}}
def study(dest,seed):
    std=list(rows(SFILE,seed,False))
    random_train=list(rows(RFILE,seed,False))
    random_test=list(rows(RFILE,seed,True))
    rng=random.Random(seed)
    train_n=min(len(std),len(random_train))
    rng.shuffle(std)
    rng.shuffle(random_train)
    std=std[:train_n]
    random_train=random_train[:train_n]
    std_score,std_p,std_counts=fitted(std)
    rand_score,rand_p,rand_counts=fitted(random_train)
    # Frozen std-train head definition
    head=set(item for item,n in std_counts.most_common(max(1,int(len(std_counts)*.2))))
    rs={
        'standard_log_item_ctr':result(random_test,std_score,std_p,head),
        'random_log_item_ctr':result(random_test,rand_score,rand_p,head),
        'constant_random_base_rate':result(random_test,{},rand_p,head),
    }
    output=Path(dest);output.mkdir(parents=True,exist_ok=False)
    meta={'status':'REAL_RANDOM_REFERENCE_PREDICTIVE_TRANSFER_NOT_CAUSAL',
         'dataset':'KuaiRand-Pure', 'seed':seed,'scenario':'tab=1 only',
         'train_arms':['same-period standard exposures','same-period randomized insertion exposures'],
         'training_users':'hash-based 80% distinct from 20% test users',
         'train_items_each':train_n,
         'random_evaluation_events':len(random_test),
         'train_std_ctr':std_p,'train_rand_ctr':rand_p,
         'test_rand_ctr':sum(x[1] for x in random_test)/len(random_test),
         'fixed_groups':'head 20% exposed item IDs by STANDARD training frequency',
         'target_scope':'Brier conditional on random insertion log, not unbiased counterfactual all-user policy value',
         'limitation':'Different logging regimes shift user/content distribution and online rewards; user heldout does not remove all domain shift',
         'results':rs,'qwen_calls':0,'confirmed_scientific_debias':False}
    (output/'results.json').write_text(json.dumps(meta,indent=2))
    print('KUAIRAND_TRANSFER',json.dumps({'seed':seed,'training':train_n,
        'n_eval_random':len(random_test),'train_ctr':(std_p,rand_p),
        'test_ctr':meta['test_rand_ctr'],
        'brier':{k:v['Brier'] for k,v in rs.items()},
        'brier_head':{k:v['brier_by_group'] for k,v in rs.items()}}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--output',required=True)
 p.add_argument('--seed',type=int,default=20261022)
 a=p.parse_args();study(a.output,a.seed)
