import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from rsi_researcher.paths import ROOT
from rsi_researcher.runtime.client import BudgetExceeded, InvalidOutput
from rsi_researcher.experiments.pilot import run_pilot, supervise
from tests.test_improvement_workflow import ProtocolModel


class PilotTests(unittest.TestCase):
    def test_external_timeout_stops_real_child_before_its_late_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            script = p / "sleeper.py"
            script.write_text("import time\nfrom pathlib import Path\ntime.sleep(5)\nPath(__file__).with_suffix('.done').write_text('late')\n")
            started = time.monotonic()
            result = supervise([sys.executable, str(script)], 0.1, p / "log")
            self.assertEqual(result["status"], "timed_out")
            self.assertLess(time.monotonic() - started, 4)
            self.assertFalse(script.with_suffix(".done").exists())

    def test_worker_pairs_all_fixed_worlds_and_preserves_initial_state(self):
        protocol = {"seeds": [11, 12], "strata": ["unrelated", "aligned"], "users": 1, "items": 12, "events_per_user": 1}
        prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = run_pilot(p, protocol, prompts, ProtocolModel(budget=100))
            self.assertEqual(len(rows), 4)
            for row in rows:
                world_dir = p / row["world"]
                initial = json.loads((world_dir / "initial/state.json").read_text())
                world = json.loads((world_dir / "world.json").read_text())
                self.assertEqual(initial, world["public"]["initial"])
                self.assertEqual(row["arms"]["original_update"]["learning_audit"]["queries_with_observed_comparison"], 0)
            self.assertEqual(json.loads((p / "pilot_summary.json").read_text())["independent_seeds"], 2)

    def test_resume_reuses_committed_events_and_rankings_without_new_calls(self):
        protocol = {"seeds": [11], "strata": ["unrelated"], "users": 1, "items": 12, "events_per_user": 2}
        prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with self.assertRaises(BudgetExceeded):
                run_pilot(p, protocol, prompts, ProtocolModel(budget=9))
            first_event = p / "11-unrelated/original_update/train-0000.json"
            committed = first_event.read_bytes()
            model = ProtocolModel(budget=100)
            run_pilot(p, protocol, prompts, model, resume=True)
            self.assertEqual(first_event.read_bytes(), committed)
            self.assertFalse(any(context.get("event") == "train-0000" for _, context in model.calls))
            self.assertEqual(sum(context["stage"] == "rank" for _, context in model.calls), 2)
            completed = ProtocolModel(budget=1)
            run_pilot(p, protocol, prompts, completed, resume=True)
            self.assertEqual(completed.requests, 0)

    def test_format_failed_world_is_reported_and_does_not_prevent_later_worlds(self):
        class OneWorldFails(ProtocolModel):
            def ask(self, prompt, schema, context):
                if self.scope["world"] == "11-unrelated" and context["stage"] == "select":
                    raise InvalidOutput("test format failure")
                return super().ask(prompt, schema, context)
        protocol = {"seeds": [11, 12], "strata": ["unrelated", "aligned"], "users": 1, "items": 12, "events_per_user": 1}
        prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = run_pilot(p, protocol, prompts, OneWorldFails(budget=100))
            self.assertEqual(len(rows), 3)
            summary = json.loads((p / "pilot_summary.json").read_text())
            self.assertEqual(summary["attempted_worlds"], 4)
            self.assertEqual(len(summary["failed_worlds"]), 1)
            self.assertFalse((p / "11-unrelated/original_update/state.json").exists())
            model = OneWorldFails(budget=1)
            self.assertEqual(len(run_pilot(p, protocol, prompts, model, resume=True)), 3)
            self.assertEqual(model.requests, 0)


if __name__ == "__main__":
    unittest.main()
