"""Evidence-only memory: fresh cohort mechanisms, no Qwen self-assessment."""
from datetime import datetime,timezone
from dataclasses import asdict
import json
from rsi_researcher.research import fresh_predictions as f
from rsi_researcher.research.fresh_audit import audit
from rsi_researcher.research.memory import ResearchMemory,validate_card


def build():
    report=audit()
    if report["status"]!="INDEPENDENT_NEW_STATE_FORECAST_AUDIT_COMPLETE":
        raise ValueError("Evidence audit absent")
    source=f.NEW/"measured_numeric_evidence.json"
    sha=f.digest(source)
    cards=[]
    for task,panels in report["numeric_measured"].items():
        action=f.TASKS[task]["experiment"]
        a,b=panels["A"],panels["B"]
        metric=f.TASKS[task]["primary_metric"]
        values=[round(panels[c][metric],5) for c in ("A","B")]
        quality=[panels[c]["quality_pass"] for c in ("A","B")]
        card=ResearchMemory(
          card_id="fresh-agentcf-20261017-"+task,
          task_id="fresh-agentcf-20261017-"+task,
          created_utc=datetime.now(timezone.utc).isoformat(),
          source_file=str(source),source_sha256=sha,
          mechanism_tags=["agentcf",task,"memory" if task=="item_memory" else "popularity"],
          decision_action=action,
          conditional_lesson=(f"New disjoint 6-user AgentCF development cohort, two panels of three: "
             f"{metric}={values}; NDCG 99% safety gates={quality}. "
             f"Qwen structured forecast did not pass both panels; do not infer mechanism absence."),
          limitation=("Single training state with six known validation targets; "
                      "identical author source data, incomplete relevance, no independent confirmation."),
          evidence_grade="development_only",split="development",confirmed=False)
        validate_card(card)
        cards.append(asdict(card))
    return {"status":"VERIFIED_NUMERIC_DEVELOPMENT_ONLY",
            "source_sha256":sha,"confirmed":0,
            "qwen_free_text_used":False,"cards":cards}


def main():
    data=build()
    path=f.NEW/"SOURCE_CHECKED_RESEARCH_MEMORY.json"
    with path.open("x",encoding="utf8") as file:
        json.dump(data,file,ensure_ascii=False,indent=2,sort_keys=True)
        file.write("\n")
    print(json.dumps({"status":data["status"],"cards":len(data["cards"]),
            "confirmed":0},ensure_ascii=False))


if __name__=="__main__":
    main()
