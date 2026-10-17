"""Cross-task Research Memory with temporal and provenance separation.

Unlike v2's within-context memory filter, this module admits VERIFIED prior
evidence even when the prior source file is not among the current task files.
No data is copied from holdout or confirmation tasks into planner context.
"""
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import json,re
from dataclasses import dataclass,asdict
from typing import Iterable

BASE=Path("/data/cjh/rsi").resolve()
TOKEN_RE=re.compile(r"[a-z0-9]+")
def tokens(s):
    return set(TOKEN_RE.findall(str(s).lower().replace("_"," ")))
def timestamp(x):
    s=str(x)
    t=datetime.fromisoformat(s.replace("Z","+00:00"))
    if t.tzinfo is None:raise ValueError("naive time")
    return t.astimezone(timezone.utc)
def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
@dataclass(frozen=True)
class ResearchMemory:
    card_id:str
    task_id:str
    created_utc:str
    source_file:str
    source_sha256:str
    mechanism_tags:list[str]
    decision_action:str
    conditional_lesson:str
    limitation:str
    evidence_grade:str
    split:str="development"
    confirmed:bool=False

def validate_card(m:ResearchMemory):
    if m.split!="development":
        raise ValueError("Only development history is eligible")
    if m.evidence_grade not in ("development_only","conditional","independently_confirmed"):
        raise ValueError("Unknown evidence grade")
    if m.confirmed != (m.evidence_grade=="independently_confirmed"):
        raise ValueError("Confirmed flag contradicts grade")
    if not all((m.card_id,m.task_id,m.mechanism_tags,m.conditional_lesson,m.limitation)):
        raise ValueError("Missing memory fields")
    timestamp(m.created_utc)
    p=Path(m.source_file).resolve()
    if not p.is_relative_to(BASE) or not p.is_file():
        raise ValueError("History evidence outside authorized project")
    if sha(p)!=m.source_sha256:raise ValueError("History evidence hash mismatch")

def retrieve_history(*,task_id:str,mechanism:str,started_utc:str,
    cards:Iterable[ResearchMemory],limit:int=3,excluded_source_hashes=()):
    """Return only traceable, prior, non-current, semantically related cards."""
    cutoff=timestamp(started_utc)
    target=tokens(mechanism)
    forbidden=set(excluded_source_hashes)
    out=[]
    for m in cards:
        try:validate_card(m)
        except (ValueError,OSError):continue
        if m.task_id==task_id or timestamp(m.created_utc)>=cutoff:continue
        if m.source_sha256 in forbidden:continue
        overlap=len(target & set().union(*(tokens(s) for s in m.mechanism_tags)))
        if not overlap:continue
        score=overlap*10+(2 if m.confirmed else 0)
        out.append((score,m.card_id,m))
    out.sort(key=lambda q:(-q[0],q[1]))
    return [{
        "card_id":m.card_id,"source_sha256":m.source_sha256,
        "source_file":m.source_file,"evidence_grade":m.evidence_grade,
        "conditional_lesson":m.conditional_lesson,"limitation":m.limitation,
        "decision_action":m.decision_action,
        "reason":"prior verified source, mechanism overlap and development partition"
        } for _,_,m in out[:max(0,limit)]]

def build_prior_cards():
    """Seed prior cards ONLY from the already committed completed development diagnostics."""
    run=BASE/"outputs/research-planner-v2-20261016/runs/source-isolated-v2"
    episodes=run/"cpu_research_episodes.json"
    situations=run/"evidence_situations.json"
    if not episodes.is_file() or not situations.is_file():raise FileNotFoundError("Prior run missing")
    from rsi_researcher.research.planner_adapters import load_real_situations
    cards=[]
    rows=json.loads(episodes.read_text(encoding="utf8"))
    sources={v["situation_id"]:v for v in json.loads(situations.read_text(encoding="utf8"))["situations"]}
    current_sources={s.situation_id:s for s in load_real_situations()}
    prior_date=json.loads((run/"status.json").read_text(encoding="utf8"))["timestamp_utc"]
    for i,e in enumerate(rows):
        sid=e["situation_id"]
        if sid not in sources:raise ValueError("untraceable historic task")
        source=sources[sid]
        source_context=current_sources[sid]
        first=next(iter(source_context.evidence.values()))
        assert first.sha256 in source["source_fingerprints"].values(), "Historical evidence source changed"
        # Preserve the distinct source file/sha for each prior task; not a common episode log hash.
        cards.append(ResearchMemory(
          card_id="prior-"+sid,task_id=sid,created_utc=prior_date,
          source_file=first.source_file,source_sha256=first.sha256,
          mechanism_tags=source["mechanism"].replace("_"," ").split(),
          decision_action=e["policy"]["next_action"],
          conditional_lesson=e["cpu_measurement"]["scientific_conclusion"],
          limitation="Previous development task; not a held-out or causal confirmation.",
          evidence_grade="development_only"))
    for c in cards:validate_card(c)
    return cards
