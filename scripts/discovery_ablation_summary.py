"""Summarize equal-budget Discovery runs; counts are not scientific validation."""
from pathlib import Path
import collections
import json
import sys


def summary(path: Path):
    final = json.loads((path / "frozen_candidate.json").read_text())
    journal = json.loads((path / "journal.json").read_text())
    started, used_tokens = 0, 0
    log = path / "model_calls" / "calls.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            item = json.loads(line)
            if item.get("status") == "started":
                started += 1
            elif item.get("status") == "received":
                used_tokens += int((item.get("usage") or {}).get("total_tokens") or 0)
    protocol_path = path / "discovery_protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    return {"path": str(path), "protocol": {
                "observation_cards_enabled": protocol.get("observation_cards_enabled"),
                "compact_experiment_enabled": protocol.get("compact_experiment_enabled")},
            "status": final.get("status"), "llm_requests_started": started,
            "tokens_reported": used_tokens,
            "successful_workbench": sum("experiment" in e for e in journal),
            "executed_countertests": sum("critic_experiment" in e for e in journal),
            "revisions": sum("revised_concept" in e for e in journal),
            "tool_errors": sum("tool_error" in e for e in journal),
            "funnel": final.get("research_funnel", {}),
            "independent_confirmation": final.get("independent_confirmation", False)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: discovery_ablation_summary.py RUN_DIRECTORY ...")
    for arg in sys.argv[1:]:
        print(json.dumps(summary(Path(arg)), ensure_ascii=False))
