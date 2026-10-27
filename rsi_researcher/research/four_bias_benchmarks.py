"""Frozen, identifiable dev/confirmation panels for Selection and Exposure.

No confirmation outcome is read until phase='confirm'. These benchmarks
describe distinct estimands: correcting an imposed MNAR mechanism vs
estimating a static click-policy value with randomized logged actions.
"""
from __future__ import annotations
import csv, hashlib, io, json, math, zipfile
from pathlib import Path
import numpy as np

ROOT=Path("/data/cjh/rsi/project")
KUAIREC=ROOT/"data/external/kuairec/KuaiRec.zip"
OBD=ROOT/"data/external/open-bandit-demo/open_bandit_dataset_full.zip"
BTS=ROOT/"data/external/open-bandit-demo/bts_all.csv"
SEEDS=(20261029,20261030,20261031)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def user_bucket(uid):
    # Stable cohort, identical across all search seeds and research methods.
    n=int.from_bytes(hashlib.sha256(f"science-final-v1:{uid}".encode()).digest()[:4],"big")%10
    return "train" if n<7 else "dev" if n<9 else "confirm"

def ci95(samples,seed=20261022,draws=400):
    x=np.asarray(samples,dtype=float)
    if len(x)<12:return None
    rng=np.random.default_rng(seed)
    means=np.asarray([np.mean(x[rng.integers(0,len(x),size=len(x))])
                      for _ in range(draws)])
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]

class SelectionPanel:
    identification="known_extra_MNAR_mask_on_near_full_KuaiRec"
    def __init__(self,phase="dev"):
        if phase not in ("dev","confirm"):raise ValueError(phase)
        self.phase=phase
        train_item=[];train_label=[];groups={}
        panel_user=[];panel_item=[];panel_label=[]
        with zipfile.ZipFile(KUAIREC) as z:
            with z.open("KuaiRec 2.0/data/small_matrix.csv") as f:
                reader=csv.DictReader(io.TextIOWrapper(f,encoding="utf-8"))
                for row in reader:
                    user=int(row["user_id"])
                    bucket=groups.get(user)
                    if bucket is None:
                        bucket=groups[user]=user_bucket(user)
                    if bucket=="train":
                        train_item.append(int(row["video_id"]))
                        train_label.append(int(float(row["watch_ratio"])>=1))
                    elif bucket==phase:
                        panel_user.append(user)
                        panel_item.append(int(row["video_id"]))
                        panel_label.append(int(float(row["watch_ratio"])>=1))
        self.item=np.asarray(train_item,dtype=np.int32)
        self.y=np.asarray(train_label,dtype=np.uint8)
        self.u=np.asarray(panel_user,dtype=np.int32)
        self.v=np.asarray(panel_item,dtype=np.int32)
        self.truth=np.asarray(panel_label,dtype=np.uint8)
        self.nitems=int(max(self.item.max(),self.v.max())+1)
        self.count=np.bincount(self.item,minlength=self.nitems)
        self.positive=np.bincount(self.item,weights=self.y,minlength=self.nitems)
        self.catalog=np.flatnonzero(self.count>0)
        ordered=sorted(self.catalog,key=lambda i:(self.positive[i]/self.count[i],i))
        head=set(ordered[int(.8*len(ordered)):])
        self.head=np.zeros(self.nitems,dtype=bool)
        self.head[list(head)]=True
        self.users=np.unique(self.u)
        self.indices={int(u):np.flatnonzero(self.u==u) for u in self.users}
        self.cohort_count={bucket:sum(v==bucket for v in groups.values())
                           for bucket in ("train","dev","confirm")}
        # A confirmation-only instance does not contain development labels and vice versa.
        self.mask_cache={}
    def scores(self,params,seed):
        blend=float(params["blend"]);cap=float(params["clip"])
        if not(0<=blend<=1 and 1<=cap<=5):raise ValueError("illegal MNAR strategy")
        if seed not in self.mask_cache:
            rng=np.random.default_rng(seed)
            prob=np.where(self.head[self.item],
                np.where(self.y==1,.85,.20),
                np.where(self.y==1,.35,.75))
            chosen=rng.random(len(self.y))<prob
            self.mask_cache[seed]=(chosen,prob)
        picked,prob=self.mask_cache[seed]
        item=self.item[picked];label=self.y[picked]
        nn=np.bincount(item,minlength=self.nitems)
        pp=np.bincount(item,weights=label,minlength=self.nitems)
        prior=label.mean()
        naive=(pp+25*prior)/(nn+25)
        w=np.minimum(1/prob[picked],cap)
        nw=np.bincount(item,weights=w,minlength=self.nitems)
        pw=np.bincount(item,weights=w*label,minlength=self.nitems)
        weighted_prior=pw.sum()/nw.sum()
        ipw=(pw+25*weighted_prior)/(nw+25)
        return (1-blend)*naive+blend*ipw
    def measure(self,params,seed,*,details=False):
        score=self.scores(params,seed)
        pred=score[self.v]
        errors=(pred-self.truth)**2
        by_user={}
        top=self.catalog[np.argsort(-score[self.catalog],kind="stable")[:50]]
        rank_positions={int(item):i for i,item in enumerate(top)}
        ndcgs=[]
        for user,ix in self.indices.items():
            labels=dict(zip(self.v[ix].tolist(),self.truth[ix].tolist()))
            positives=sum(labels.values())
            dcg=sum(1/math.log2(r+2) for item,r in rank_positions.items()
                    if labels.get(item,0))
            ideal=sum(1/math.log2(j+2) for j in range(min(50,positives)))
            nd=dcg/ideal if ideal else 0.
            ndcgs.append(nd)
            by_user[user]={"Brier":float(np.mean(errors[ix])),"NDCG@50":nd,
                           "positives":int(positives)}
        metric={"Brier":float(errors.mean()),"NDCG@50":float(np.mean(ndcgs)),
                "events":len(self.v),"users":len(self.users),
                "observed_protocol":"train-only extra known MNAR; independent user reference",
                "effective_sample_size":len(self.users)}
        return (metric,by_user) if details else metric
    def compare(self,baseline,candidate,seed):
        a,aa=self.measure(baseline,seed,details=True)
        b,bb=self.measure(candidate,seed,details=True)
        ids=sorted(set(aa)&set(bb))
        positive=np.array([aa[i]["Brier"]-bb[i]["Brier"] for i in ids])
        quality=np.array([bb[i]["NDCG@50"]-aa[i]["NDCG@50"] for i in ids])
        return {"base":a,"candidate":b,"delta_primary":a["Brier"]-b["Brier"],
                "CI95_gain":ci95(positive,seed),
                "CI95_utility_delta":ci95(quality,seed+10),
                "user_blocks":len(ids)}

class ExposurePanel:
    identification="known_random_logger_propensity"
    def __init__(self,phase="dev"):
        if phase not in ("dev","confirm"):raise ValueError(phase)
        self.phase=phase
        self.count=np.zeros((3,80),dtype=np.int64)
        with BTS.open(newline="") as f:
            for r in csv.DictReader(f):
                i=int(r["item_id"]);pos=int(r["position"])-1
                if 0<=i<80 and 0<=pos<3:self.count[pos,i]+=1
        self.ranks=[np.argsort(-self.count[p],kind="stable") for p in range(3)]
        # Full 3.6M random-reference events, the 10k demo is NOT reused as evaluator.
        action=[];position=[];reward=[];pscore=[];blocks=[]
        with zipfile.ZipFile(OBD) as z:
            with z.open("open_bandit_dataset/random/all/all.csv") as f:
                rdr=csv.reader(io.TextIOWrapper(f,encoding="utf-8"))
                hdr=next(rdr)
                ia,ip,ir,ips=[hdr.index(k) for k in
                             ("item_id","position","click","propensity_score")]
                for ix,r in enumerate(rdr):
                    # Exclude the public 10k sample already explored in older
                    # runs; preserve a larger unseen full-data reference.
                    if ix<10000:continue
                    # Permanently frozen 80/20 disjoint impression partition.
                    holdout=(ix%5==4)
                    if holdout!=(phase=="confirm"):continue
                    a=int(r[ia]);p=int(r[ip])-1;y=int(r[ir]);prop=float(r[ips])
                    if not (0<=a<80 and 0<=p<3 and abs(prop-.0125)<1e-8):
                        raise ValueError("random logger action/propensity mismatch")
                    action.append(a);position.append(p);reward.append(y)
                    pscore.append(prop)
                    blocks.append(ix//500)
        self.action=np.asarray(action,dtype=np.int16)
        self.pos=np.asarray(position,dtype=np.int8)
        self.y=np.asarray(reward,dtype=np.uint8)
        self.prop=np.asarray(pscore,dtype=np.float32)
        self.block=np.asarray(blocks,dtype=np.int32)
        self.n=len(self.y)
        self.total_clicks=int(self.y.sum())
        # Shared BTS demo head ranking pre-frozen from a different logging policy.
    def weights(self,params):
        top=int(params["top"]);mass=float(params["mass"])
        if not(5<=top<=40 and .25<=mass<=1):raise ValueError("illegal static target policy")
        probs=np.full((3,80),(1-mass)/(80-top),dtype=float)
        for p in range(3):probs[p,self.ranks[p][:top]]=mass/top
        return probs[self.pos,self.action]/self.prop
    def measure(self,params,seed=0,*,details=False):
        # Three disjoint groups of consecutive time blocks, not 3 repeated
        # evaluations of the same development impressions.
        fold=SEEDS.index(seed) if seed in SEEDS else int(seed)%3
        selected=self.block%3==fold
        weights=self.weights(params)[selected]
        y=self.y[selected]
        ess=float(weights.sum()**2/np.square(weights).sum())
        ips=float(np.mean(weights*y))
        snips=float(np.sum(weights*y)/np.sum(weights))
        m={"SNIPS":snips,"IPS":ips,"ESS":ess,
            "ESS_fraction":ess/len(y),"events":int(len(y)),"clicks":int(y.sum()),
           "weight_mean":float(np.mean(weights)),
           "observed_protocol":"known-random-logger OPE; not exposure bias elimination"}
        return (m,weights) if details else m
    def compare(self,baseline,candidate,seed):
        a,wa=self.measure(baseline,seed,details=True)
        b,wb=self.measure(candidate,seed,details=True)
        # Paired cluster bootstrap by consecutive 500-row time blocks;
        # reused dev events across 3 seeds are NOT independent repetitions.
        fold=SEEDS.index(seed) if seed in SEEDS else int(seed)%3
        selected=self.block%3==fold
        block=self.block[selected]
        ix=np.unique(block,return_inverse=True)[1]
        n=int(ix.max()+1)
        y=self.y[selected]
        ar=np.bincount(ix,weights=wa*y,minlength=n)
        br=np.bincount(ix,weights=wb*y,minlength=n)
        aw=np.bincount(ix,weights=wa,minlength=n)
        bw=np.bincount(ix,weights=wb,minlength=n)
        rng=np.random.default_rng(seed)
        gains=[]
        for _ in range(400):
            chosen=rng.integers(0,n,n)
            gains.append(br[chosen].sum()/bw[chosen].sum() -
                         ar[chosen].sum()/aw[chosen].sum())
        gain_ci=[float(np.quantile(gains,.025)),float(np.quantile(gains,.975))]
        return {"base":a,"candidate":b,"delta_primary":b["SNIPS"]-a["SNIPS"],
                "CI95_gain":gain_ci,"time_blocks":n,
                 "min_ESS_required":max(250,int(.1*len(y)))}
