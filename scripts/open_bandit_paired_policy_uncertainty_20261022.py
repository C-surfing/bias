"""Paired Bootstrap for OBD OPE policy value differences, same random-log events."""
from collections import Counter,defaultdict
from pathlib import Path
import csv,random,json
data=Path("/data/cjh/rsi/project/data/external/open-bandit-demo")
root=Path("/data/cjh/rsi/project/runs/four-bias-exposure-agent-workflow-20261022-v3")
with (data/"bts_all.csv").open() as f:
    bts=list(csv.DictReader(f))
with (data/"random_all.csv").open() as f:
    obs=list(csv.DictReader(f))
counts=defaultdict(Counter)
for r in bts:counts[int(r["position"])][int(r["item_id"])]+=1
head={pos:set(sorted(range(80),key=lambda i:(-counts[pos][i],i))[:20]) for pos in (1,2,3)}
rows=[]
for r in obs:
    pos=int(r["position"]);itm=int(r["item_id"]);reward=int(r["click"])
    is_head=itm in head[pos]
    rows.append((reward,4.0*is_head,3.7 if is_head else .1))
def snips(sample,which):
    denom=sum(r[which] for r in sample)
    return sum(r[0]*r[which] for r in sample)/denom
observed={"head20_minus_fixed":snips(rows,1)-snips(rows,2),
          "head20_minus_uniform":snips(rows,1)-sum(r[0] for r in rows)/len(rows)}
rng=random.Random(20261022)
samples={k:[] for k in observed}
for _ in range(600):
    ids=[rng.randrange(len(rows)) for _ in range(len(rows))]
    s=[rows[i] for i in ids]
    x=snips(s,1)
    samples["head20_minus_fixed"].append(x-snips(s,2))
    samples["head20_minus_uniform"].append(x-sum(r[0] for r in s)/len(s))
ci={k:sorted(v) for k,v in samples.items()}
result={"source":"Open Bandit 10k randomized reference same-user-event paired bootstrap",
  "observed_delta":observed,
  "CI95":{k:[v[int(len(v)*.025)],v[int(len(v)*.975)]] for k,v in ci.items()},
  "draws":600,"n":len(rows),"independent_holdout_confirmation":False,
  "method":"same logged samples, paired policy importance weights, not equal-support OPE causal proof"}
(root/"paired_uncertainty.json").write_text(json.dumps(result,indent=2))
print("EXPOSURE_PAIRED_CI",json.dumps(result),flush=True)
