"""Persistent research history shared by discovery and independent validation."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from rsi_researcher.runtime.client import atomic_json


def _now():
    return datetime.now(timezone.utc).isoformat()


class ExperimentJournal:
    """Keep trial outcomes and their later validation feedback in one JSON file."""

    def __init__(self, path):
        self.path = Path(path).resolve()

    def _load(self):
        if not self.path.exists():
            return {"version": "research_journal_v1", "entries": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _relative(self, path):
        return Path(os.path.relpath(Path(path).resolve(), self.path.parent)).as_posix()

    def _write(self, document):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, document)

    def entries(self):
        return deepcopy(self._load()["entries"])

    def record_trial(self, trial_dir, archive):
        """Record successful and unsuccessful reasoning, without copying worlds."""
        trial_path = self._relative(trial_dir)
        saved = deepcopy(archive)
        # The archive is the researcher-visible record, not an evaluator world.
        saved.pop("world", None)
        saved.pop("private", None)
        if saved.get("baseline_source"):
            saved["baseline_source"] = self._relative(saved["baseline_source"])
        execution = saved.get("execution") or {}
        identity = execution.get("experiment_id") or hashlib.sha256(
            json.dumps({"trial_path": trial_path, "proposal": saved.get("proposal")},
                       ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        candidate_id = "candidate-" + identity
        entry = {"kind": "trial", "candidate_id": candidate_id, "trial_path": trial_path,
                 "recorded_at": _now(), "archive": saved}
        document = self._load()
        records = document["entries"]
        for index, old in enumerate(records):
            if old["kind"] == "trial" and old["trial_path"] == trial_path:
                records[index] = entry
                break
        else:
            records.append(entry)
        self._write(document)
        return candidate_id

    def record_validation(self, candidate_id, result):
        """Attach external results to the candidate, including failed conditions."""
        document = self._load()
        if not any(e["kind"] == "trial" and e["candidate_id"] == candidate_id
                   for e in document["entries"]):
            raise ValueError("validation candidate is absent from the experiment journal")
        saved = deepcopy(result)
        saved.pop("world", None)
        saved.pop("private", None)
        for key in ("directory", "source_trial"):
            if saved.get(key):
                saved[key] = self._relative(saved[key])
        document["entries"].append({"kind": "validation", "candidate_id": candidate_id,
                                    "recorded_at": _now(), "result": saved})
        self._write(document)

    def research_history(self):
        """Return existing archive-shaped records with all later validation feedback."""
        entries = self.entries()
        feedback = {}
        for entry in entries:
            if entry["kind"] == "validation":
                feedback.setdefault(entry["candidate_id"], []).append(entry["result"])
        result = []
        for entry in entries:
            if entry["kind"] != "trial":
                continue
            record = entry["archive"]
            record["candidate_id"] = entry["candidate_id"]
            record["trial_path"] = entry["trial_path"]
            record["validation_feedback"] = deepcopy(feedback.get(entry["candidate_id"], []))
            result.append(record)
        return result
