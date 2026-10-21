"""Read-only numerical experiment report; does not select on holdout."""
import argparse
import json
from pathlib import Path

METRICS=("HR@5","NDCG@5","ARP@5","APLT@5","CandidatePositionBias_NDCG@5")

def number(v):
    return "N/A" if v is None else f"{v:.6f}"

def show(path):
    data=json.loads((path/"result.json").read_text(encoding="utf-8"))
    before=data["development"]["baseline"]["5"]
    rows=[]
    def row(name,stat,audit=None):
        vals=[number(stat["5"].get(k)) for k in METRICS]
        rows.append((name,*vals))
        print(name,"|"," | ".join(vals),
              "gate="+str((audit or {}).get("eligible_development_only","n/a")))
    print("Source:",path)
    print("Phase:","HR5 | NDCG5 | ARP5 | APLT5 | CandidatePositionBias_NDCG5")
    row("DEV baseline",data["development"]["baseline"])
    for f in data["development"]["fixed"]:
        row("DEV "+f["name"],f["metrics"],f["audit"])
    for t in data["development"]["tree"]:
        if t["status"]=="MEASURED":
            row("DEV "+t["id"],t["metrics"],t.get("audit"))
        elif t["status"]=="REJECTED":
            print("DEV",t["id"],"critic_rejected")
    for arm in ("baseline","fixed","rsi"):
        row("HOLDOUT "+arm,data["holdout"][arm])
    print("Selections:",data["development"]["selected_rsi"]["id"],
          data["development"]["selected_fixed"]["name"])
    print("Paired 95% NDCG CI",data["ci95"]["rsi_minus_baseline"]["NDCG@5"])
    return rows

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run_directory",type=Path)
    a=p.parse_args()
    show(a.run_directory)
