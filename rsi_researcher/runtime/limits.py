"""Persisted client-side quotas; token reservation is an estimate, not provider billing."""

import json
from pathlib import Path
import time


class RunLimits:
    def __init__(self, directory, max_requests, max_tokens, max_seconds):
        from rsi_researcher.runtime.client import atomic_json
        if any(type(v) is not int or v <= 0 for v in (max_requests, max_tokens, max_seconds)):
            raise ValueError("limits must be positive integers")
        path = Path(directory) / "limits.json"
        config = {"max_requests": max_requests, "max_tokens": max_tokens, "max_seconds": max_seconds}
        log = Path(directory) / "calls.jsonl"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["config"] != config:
                raise ValueError("cannot change a run's frozen limits")
        else:
            if log.exists():
                raise ValueError("cannot attach new quotas to a prior unbounded run")
            saved = {"config": config, "deadline_epoch": time.time() + max_seconds,
                     "reservation": "UTF-8 request bytes + 1024 framing allowance + max output tokens"}
            atomic_json(path, saved)
        self.max_requests, self.max_tokens = max_requests, max_tokens
        self.deadline_epoch = saved["deadline_epoch"]
        self.deadline_monotonic = time.monotonic() + max(0, self.deadline_epoch - time.time())
        previous = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        charges = {}
        for row in previous:
            if row["status"] == "started":
                charges[row["call_id"]] = row["reserved_tokens"]
            elif row["status"] == "received":
                charges[row["call_id"]] = self.reported_tokens(row.get("usage"), charges[row["call_id"]])
        self.charged_tokens = sum(charges.values())

    @staticmethod
    def reported_tokens(usage, fallback):
        if not isinstance(usage, dict):
            return fallback
        total = usage.get("total_tokens")
        if type(total) is int and total >= 0:
            return total
        values = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
        return sum(values) if all(type(v) is int and v >= 0 for v in values) else fallback

    def remaining_seconds(self):
        from rsi_researcher.runtime.client import BudgetExceeded
        remaining = min(self.deadline_epoch - time.time(), self.deadline_monotonic - time.monotonic())
        if remaining <= 0:
            raise BudgetExceeded("run time allowance exhausted")
        return remaining

    def reserve(self, payload):
        from rsi_researcher.runtime.client import BudgetExceeded
        self.remaining_seconds()
        estimate = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 1024 + payload["max_tokens"]
        if self.charged_tokens + estimate > self.max_tokens:
            raise BudgetExceeded("insufficient token allowance for next request")
        self.charged_tokens += estimate
        return estimate

    def settle(self, reservation, usage):
        self.charged_tokens += self.reported_tokens(usage, reservation) - reservation
        from rsi_researcher.runtime.client import BudgetExceeded
        self.remaining_seconds()
        if self.charged_tokens > self.max_tokens:
            raise BudgetExceeded("reported token usage exceeded reservation estimate")
