"""KuaiRand-Pure: REAL randomized/standard impression observational audit.
Never infer logging propensities or causal policy values from a random-labeled file.
"""
import argparse,csv,hashlib,json,math,random
from collections import Counter,defaultdict
from pathlib import Path

BASE=Path('/data/cjh/rsi/project/data/external/kuairand-pure/KuaiRand-Pure/data')
FILES={'train':'log_standard_4_08_to_4_21_pure.csv',
       'random':'log_random_4_22_to_5_08_pure.csv',
       'standard':'log_standard_4_22_to_5_08_pure.csv'}

def iter_rows(path):
    with path.open(newline='') as file:
        yield from csv.DictReader(file)

def wilson(k,n,z=1.96):
    if n==0:return None
    p=k/n
    a=1+z*z/n
    m=(p+z*z/(2*n))/a
    e=z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/a
    return [m-e,m+e]

def aggregate(path,head,pool):
    stats=defaultdict(lambda:Counter())
    usr=defaultdict(set)
    videos=defaultdict(set)
    for r in iter_rows(path):
        if r['video_id'] not in pool:continue
        group='head' if r['video_id'] in head else 'nonhead'
        tab=r['tab']
        c=int(r['is_click'])
        for key in (('all',group),('tab_'+tab,group)):
            s=stats[key];s['impressions']+=1;s['clicks']+=c
            usr[key].add(r['user_id']);videos[key].add(r['video_id'])
        for key in (('all','total'),('tab_'+tab,'total')):
            stats[key]['impressions']+=1
            stats[key]['clicks']+=c
            stats[key]['is_rand_flag_one']+=int(r['is_rand']=='1')
            usr[key].add(r['user_id'])
            videos[key].add(r['video_id'])
    result={}
    for (kind,group),s in sorted(stats.items()):
        result[kind+':'+group]=dict(impressions=s['impressions'],clicks=s['clicks'],
            click_rate=s['clicks']/s['impressions'],click_rate_wilson95=wilson(s['clicks'],s['impressions']),
            unique_users=len(usr[(kind,group)]),unique_videos=len(videos[(kind,group)]),
            insertion_flag_share=(s['is_rand_flag_one']/s['impressions'] if group=='total' else None))
    return result

def run(dest):
    dest=Path(dest)
    assert not dest.exists(), 'output must be new'
    train_counts=Counter()
    train_users=set()
    for r in iter_rows(BASE/FILES['train']):
        train_counts[r['video_id']]+=1
        train_users.add(r['user_id'])
    pool=set(train_counts)
    # Head = top 20% of items observed in the PRE period; never use future labels.
    top=max(1,math.ceil(.2*len(pool)))
    head=set(sorted(pool,key=lambda x:(-train_counts[x],x))[:top])
    arms={a:aggregate(BASE/path,head,pool) for a,path in FILES.items() if a!='train'}
    unrecognized={}
    for a in ('random','standard'):
        stats=arms[a]
        for tab in sorted(set(k.split(':')[0] for k in stats)):
            tot=stats.get(tab+':total',{}).get('impressions',0)
            hh=stats.get(tab+':head',{}).get('impressions',0)
            hc=stats.get(tab+':head',{}).get('clicks',0)
            tt=stats.get(tab+':nonhead',{}).get('impressions',0)
            tc=stats.get(tab+':nonhead',{}).get('clicks',0)
            unrecognized[a+':'+tab]=dict(total_observed_impressions=tot,head_exposure_share=hh/tot if tot else None,
               head_click_rate=hc/hh if hh else None,nonhead_click_rate=tc/tt if tt else None,
               head_exposure_count=hh,nonhead_exposure_count=tt)
    dest.mkdir(parents=True)
    evidence=dict(status='REAL_LOG_OBSERVATIONAL_SUPPORT_AUDIT',dataset='KuaiRand-Pure',
      source_files=FILES,group_definition='Head top 20% of distinct PRE-period standard exposed items, others nonhead',
      pre_period_users=len(train_users),pre_period_items=len(pool),head_items=len(head),
      outcome_definition='is_click: UI-dependent click or valid play; stratify by tab',
      random_intervention_logs='Randomly inserted item exposure records, not uniform complete user-item matrix',
      identified_policy_value=False,logging_propensities_known=False,
      logging_probabilities_required_before_ips_dr=True,
      support='same time window April 22-May 8; random and standard logs have different assignment policies',
      descriptive_by_policy=unrecognized,
      detail=arms,
      limitations=['CTR between assignment policies is NOT unbiased estimated treatment effect',
                   'No unexposed negatives are imputed',
                   'Nonhead excludes items absent in PRE logs; denominator specified',
                   'Video merit and protected provider groups are NOT identified',
                   'This protocol is a data adequacy and diagnostic study, not a debias result'],
      qwen_calls=0,holdout_opened=False)
    (dest/'results.json').write_text(json.dumps(evidence,indent=2))
    print('KUAIRAND',json.dumps({'train_items':len(pool),'train_users':len(train_users),
      'arms':{a:{'all':unrecognized[a+':all'],
                 'by_tab':{k:v for k,v in unrecognized.items() if k.startswith(a+':tab_')}}
                   for a in ('random','standard')}}))
    print('DIR',dest)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--output',required=True)
 a=p.parse_args()
 run(a.output)
