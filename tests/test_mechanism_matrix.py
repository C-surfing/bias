from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from textwrap import dedent
import unittest

from rsi_researcher.runtime.client import API_PROTOCOL, MEMORY_CONTROL_PROTOCOL, MODEL_PARAMETERS, RANKING_PROTOCOL, BudgetExceeded, InvalidOutput, ModelFailure, atomic_json
from rsi_researcher.paths import ROOT
from rsi_researcher.experiments.matrix import case_id, matrix_schedule, run_matrix, summarize_matrix, validate_protocol
from rsi_researcher.datasets.controlled import WORLD_PROTOCOL
from tests.test_improvement_workflow import ProtocolModel


class MatrixModel(ProtocolModel):
    def ask(self, prompt, schema, context):
        return super().ask(prompt, schema, (getattr(self, "scope", None) or {}) | context)


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"seeds": [71], "strata": ["unrelated", "aligned"], "users": 2, "items": 12,
                         "events_per_user": 1, "order_seed": 919}
        self.prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))

    def ready_protocol(self, root):
        pilot = root / "pilot"
        pilot.mkdir()
        atomic_json(pilot / "protocol.json", {"seeds": [1, 2, 3], "strata": ["unrelated", "aligned"], "users": 12,
                    "items": 60, "events_per_user": 4, "api_protocol": API_PROTOCOL, "ranking_protocol": RANKING_PROTOCOL,
                    "model_parameters": deepcopy(MODEL_PARAMETERS),
                    "world_protocol": WORLD_PROTOCOL, "evaluation_panel_size": 20})
        atomic_json(pilot / "run_result.json", {"status": "completed"})
        atomic_json(pilot / "completed_worlds.json", [{"seed": s, "stratum": layer, "utility_gain": 0, "bias_change": 0}
                    for s in (1, 2, 3) for layer in ("unrelated", "aligned")])
        return self.protocol | {"status": "frozen_exploratory", "purpose": "E2_mechanism_only", "pilot_run": str(pilot),
                    "model_parameters": deepcopy(MODEL_PARAMETERS), "api_protocol": API_PROTOCOL, "ranking_protocol": RANKING_PROTOCOL,
                    "memory_control_protocol": MEMORY_CONTROL_PROTOCOL, "position_scope": "training_pairs_only",
                    "primary_contrast": "paired_balanced_bias_interaction", "candidate_promotion_allowed": False,
                    "world_protocol": WORLD_PROTOCOL, "evaluation_panel_size": 12,
                    "max_requests": 1000, "max_tokens": 100000, "max_seconds": 600}

    def test_schedule_has_eight_cells_paired_pools_and_fixed_ablations(self):
        schedule = matrix_schedule(self.protocol)
        self.assertEqual(len(schedule), 36)
        self.assertEqual(len({case_id(s) for s in schedule}), 36)
        self.assertEqual(schedule, matrix_schedule(self.protocol))
        self.assertNotEqual(schedule, matrix_schedule(self.protocol | {"order_seed": 920}))
        for stratum in self.protocol["strata"]:
            specs = [s for s in schedule if s["stratum"] == stratum]
            groups = {}
            for spec in specs:
                groups.setdefault(spec["cell_id"], []).append(spec)
            self.assertEqual(len(groups), 11)
            for cell, pools in groups.items():
                positions = {s["position"] for s in pools}
                self.assertIn(positions, ({"pair_0", "pair_1"}, {"head_first"}))
                if cell.startswith("ablation-"):
                    self.assertEqual(positions, {"pair_0", "pair_1"})
                    self.assertEqual({s["exposure"] for s in pools}, {"high"})
        with self.assertRaises(ValueError):
            matrix_schedule(self.protocol | {"seeds": [71, 71]})

    def test_protocol_rejects_changed_panels_and_historical_generator_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol = self.ready_protocol(Path(tmp))
            validate_protocol(protocol)
            for change in ({"world_protocol": "old"}, {"evaluation_panel_size": 60}):
                with self.assertRaises(ValueError):
                    validate_protocol(protocol | change)
            path = Path(protocol["pilot_run"]) / "protocol.json"
            pilot = json.loads(path.read_text(encoding="utf-8"))
            for field in ("world_protocol", "evaluation_panel_size", "model_parameters"):
                changed = deepcopy(pilot)
                changed.pop(field)
                atomic_json(path, changed)
                with self.assertRaises(ValueError):
                    validate_protocol(protocol)
            atomic_json(path, pilot)
            validate_protocol(protocol)

    def test_interactions_average_paired_pools_and_strata_within_seed(self):
        schedule = matrix_schedule(self.protocol)
        rows = []
        for spec in schedule:
            baseline = 0.1
            bias = baseline + 0.1 * (spec["exposure"] == "high")
            if spec["memory_policy"] == "original_update":
                bias += 0.1 + (0.2 if spec["position"] == "head_first" else 0.05) * (spec["exposure"] == "high")
            if spec["position"] == "pair_0":
                bias += 0.01
            elif spec["position"] == "pair_1":
                bias -= 0.01
            rows.append({"case": case_id(spec), "spec": spec,
                         "score": {"bias": bias, "utility": 0.8, "signed_excess": 0, "position_top1_disagreement": 0}})
        self.assertFalse(summarize_matrix(self.protocol, schedule, rows, [], "failed")["complete"])
        result = summarize_matrix(self.protocol, schedule, rows, [], "completed")
        self.assertTrue(result["complete"])
        self.assertEqual(result["independent_seed_count"], 1)
        self.assertAlmostEqual(result["paired_seed_effects"][0]["paired_balanced"], 0.05)
        self.assertAlmostEqual(result["paired_seed_effects"][0]["head_first"], 0.2)
        self.assertAlmostEqual(result["paired_seed_effects"][0]["position_interaction_difference"], 0.15)
        self.assertEqual(result["statistical_support"], "not_tested")
        partial = summarize_matrix(self.protocol, schedule, rows[1:], [])
        self.assertFalse(partial["complete"])
        self.assertIn(rows[0]["case"], partial["missing_pools"])
        with self.assertRaises(ValueError):
            summarize_matrix(self.protocol, schedule, rows + rows[:1], [])

    def test_whole_matrix_is_isolated_blinded_and_resumes_without_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = MatrixModel(budget=1000)
            summary = run_matrix(root, self.protocol, self.prompts, model)
            self.assertTrue(summary["all_pools_recorded"])
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["completed_pools"], 36)
            self.assertEqual(summary["independent_seed_count"], 1)
            for spec in matrix_schedule(self.protocol):
                state = json.loads((root / case_id(spec) / "state.json").read_text(encoding="utf-8"))
                initial = json.loads((root / f"{spec['seed']}-{spec['stratum']}" / "world.json").read_text(encoding="utf-8"))["public"]["initial"]
                self.assertEqual(len(state["completed"]), 2)
                if spec["memory_policy"] == "frozen":
                    self.assertEqual(state["users"], initial["users"])
                    self.assertEqual(state["items"], initial["items"])
                if spec["memory_policy"] == "no_sharing":
                    self.assertEqual(state["items"], initial["items"])
                    self.assertIn("private_items", state)
            for prompt, context in model.calls:
                self.assertNotIn('"utilities"', prompt)
                self.assertNotIn('"popular"', prompt)
                self.assertNotIn('"preperiod_counts"', prompt)
            resumed = MatrixModel(budget=0)
            self.assertEqual(run_matrix(root, self.protocol, self.prompts, resumed, resume=True), summary)
            self.assertEqual(resumed.requests, 0)
            with self.assertRaises(ValueError):
                run_matrix(root, self.protocol | {"order_seed": 920}, self.prompts, resumed, resume=True)
            (root / case_id(matrix_schedule(self.protocol)[0]) / "rankings.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing its recorded artifacts"):
                run_matrix(root, self.protocol, self.prompts, resumed, resume=True)
            self.assertEqual(resumed.requests, 0)

    def test_format_failure_keeps_original_denominator_and_is_not_retried_on_resume(self):
        failing = case_id(matrix_schedule(self.protocol)[0])
        class OnePoolFails(MatrixModel):
            def ask(self, prompt, schema, context):
                if self.scope["case"] == failing:
                    raise InvalidOutput("test-only formatting failure")
                return super().ask(prompt, schema, context)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_matrix(root, self.protocol, self.prompts, OnePoolFails(budget=1000))
            self.assertFalse(summary["complete"])
            self.assertEqual((summary["completed_pools"], summary["expected_pools"]), (35, 36))
            self.assertEqual(summary["failed_pools"][0]["case"], failing)
            resumed = MatrixModel(budget=0)
            self.assertEqual(run_matrix(root, self.protocol, self.prompts, resumed, resume=True), summary)
            self.assertEqual(resumed.requests, 0)

    def test_budget_interrupt_can_reuse_events_and_balance_failure_stops_first_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(BudgetExceeded):
                run_matrix(root, self.protocol, self.prompts, MatrixModel(budget=6))
            committed = {p: p.read_bytes() for p in root.rglob("train-*.json")}
            self.assertTrue(committed)
            summary = run_matrix(root, self.protocol, self.prompts, MatrixModel(budget=1000), resume=True)
            self.assertTrue(summary["all_pools_recorded"])
            for p, data in committed.items():
                self.assertEqual(p.read_bytes(), data)
        class BalanceFailsAtRank(MatrixModel):
            def ask(self, prompt, schema, context):
                result = super().ask(prompt, schema, context)
                if context["stage"] == "rank":
                    raise ModelFailure("test-only insufficient balance", http_status=402)
                return result
        with tempfile.TemporaryDirectory() as tmp:
            model = BalanceFailsAtRank(budget=1000)
            with self.assertRaises(ModelFailure) as caught:
                run_matrix(Path(tmp), self.protocol, self.prompts, model)
            self.assertEqual(caught.exception.http_status, 402)
            self.assertEqual(sum(c["stage"] == "rank" for _, c in model.calls), 1)
            self.assertFalse((Path(tmp) / "matrix_summary.json").exists())

    def test_protocol_and_cli_reject_draft_incomplete_pilot_and_reused_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = self.ready_protocol(root)
            pilot = Path(protocol["pilot_run"])
            atomic_json(pilot / "run_result.json", {"status": "failed"})
            atomic_json(pilot / "completed_worlds.json", [])
            with self.assertRaisesRegex(ValueError, "pilot is incomplete"):
                validate_protocol(protocol)
            rows = [{"seed": seed, "stratum": layer, "utility_gain": 0, "bias_change": 0}
                    for seed in (1, 2, 3) for layer in ("unrelated", "aligned")]
            atomic_json(pilot / "completed_worlds.json", rows)
            atomic_json(pilot / "run_result.json", {"status": "completed"})
            validate_protocol(protocol)
            with self.assertRaisesRegex(ValueError, "independent"):
                validate_protocol(protocol | {"seeds": [1]})
            with self.assertRaises(ValueError):
                validate_protocol(protocol | {"candidate_promotion_allowed": True})
            draft = root / "draft.json"
            atomic_json(draft, protocol | {"status": "draft"})
            output = root / "must-not-exist"
            process = subprocess.run([sys.executable, "-m", "rsi_researcher.experiments.matrix", "--output", str(output), "--protocol", str(draft)],
                                     cwd=ROOT, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(process.returncode, 0)
            self.assertFalse(output.exists())

    def test_cli_snapshots_worker_and_resumes_with_original_deadline_and_zero_extra_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bootstrap = root / "test-bootstrap"
            bootstrap.mkdir()
            (bootstrap / "sitecustomize.py").write_text(dedent('''
                import json, os, re, urllib.error, urllib.request
                class Reply:
                    def __init__(self, body): self.body = body
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                    def read(self): return self.body
                def offline_reply(request, **kwargs):
                    assert request.full_url == 'https://api.deepseek.com/beta/chat/completions'
                    payload = json.loads(request.data)
                    props = payload['tools'][0]['function']['parameters']['properties']
                    if 'selected_id' in props:
                        answer = {'selected_id':props['selected_id']['enum'][0], 'reason':'test-only reason'}
                    elif 'memory' in props:
                        answer = {'memory':'test-only user memory'}
                    elif 'first_memory' in props:
                        answer = {'first_memory':'test-only first memory','second_memory':'test-only second memory'}
                    elif 'ranking' in props:
                        if os.environ.get('MATRIX_TEST_FAIL_RANK') == '1':
                            raise urllib.error.HTTPError(request.full_url, 402, 'test-only balance failure', {}, None)
                        answer = {'ranking':re.findall(r'ID=(\\w+);', payload['messages'][1]['content'])}
                    else:
                        raise AssertionError('unexpected test-only schema')
                    choice = {'finish_reason':'tool_calls','message':{'tool_calls':[{'type':'function',
                        'function':{'name':'submit_result','arguments':json.dumps(answer)}}]}}
                    return Reply(json.dumps({'model':'test-double','id':'test-only-response','choices':[choice],
                        'usage':{'prompt_tokens':10,'completion_tokens':5}}).encode())
                urllib.request.urlopen = offline_reply
            '''), encoding="utf-8")
            protocol = self.ready_protocol(root) | {"users": 1}
            source = root / "protocol.json"
            atomic_json(source, protocol)
            output = root / "batch"
            env = os.environ | {"PYTHONPATH": str(bootstrap), "DEEPSEEK_API_KEY": "test-only", "PYTHONIOENCODING": "utf-8"}
            command = [sys.executable, "-m", "rsi_researcher.experiments.matrix", "--output", str(output)]
            result = subprocess.run(command + ["--protocol", str(source)], cwd=ROOT, env=env | {"MATRIX_TEST_FAIL_RANK": "1"}, capture_output=True, text=True, timeout=45)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads((output / "run_result.json").read_text(encoding="utf-8"))["status"], "failed")
            self.assertFalse(json.loads((output / "matrix_summary.json").read_text(encoding="utf-8"))["complete"])
            self.assertIn("HTTP 状态 402", (output / "matrix_report.md").read_text(encoding="utf-8"))
            limits = (output / "limits.json").read_bytes()
            calls = (output / "calls.jsonl").read_bytes()
            self.assertEqual(sum(json.loads(s).get("http_status") == 402 for s in calls.decode("utf-8").splitlines()), 1)
            resumed = subprocess.run(command + ["--resume"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertEqual((output / "limits.json").read_bytes(), limits)
            self.assertTrue((output / "calls.jsonl").read_bytes().startswith(calls))
            calls = (output / "calls.jsonl").read_bytes()
            self.assertTrue(json.loads((output / "matrix_summary.json").read_text(encoding="utf-8"))["complete"])
            responses = [json.loads(s) for s in calls.decode("utf-8").splitlines() if json.loads(s)["status"] == "received"]
            self.assertTrue(responses)
            self.assertTrue(all(r["response_model"] == "test-double" for r in responses))
            manifest = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("rsi_researcher/experiments/matrix.py", manifest)
            self.assertIn("rsi_researcher/recommenders/agentcf.py", manifest)
            self.assertTrue((output / "code/rsi_researcher/experiments/matrix.py").exists())
            self.assertEqual(json.loads((output / "schedule.json").read_text(encoding="utf-8")), matrix_schedule(protocol))
            atomic_json(output / "run_result.json", {"status": "failed", "reason": "test-only terminal failure after recording"})
            resumed = subprocess.run(command + ["--resume"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertEqual((output / "limits.json").read_bytes(), limits)
            self.assertEqual((output / "calls.jsonl").read_bytes(), calls)
            atomic_json(output / "run_result.json", {"status": "failed"})
            refused = subprocess.run(command + ["--resume"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(refused.returncode, 0)
            self.assertEqual((output / "calls.jsonl").read_bytes(), calls)


if __name__ == "__main__":
    unittest.main()
