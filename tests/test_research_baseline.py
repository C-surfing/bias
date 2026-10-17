from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from rsi_researcher.runtime.client import API_PROTOCOL, MEMORY_CONTROL_PROTOCOL, MODEL_PARAMETERS, RANKING_PROTOCOL
from rsi_researcher.datasets.controlled import build_world, ordered_events
from rsi_researcher.research.baseline import load_baseline, make_baseline_contract, source_prompts


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.world = build_world(83, users=2, items=20, events_per_user=2)
        self.prompts = {"user_prompt_template": "Observed feedback", "system_prompt_template": "Select a CD"}

    def fixture(self, directory, pilot=False):
        root = Path(directory)
        parent = root / "world" if pilot else root
        arm = parent / ("original_update" if pilot else "B0")
        save(parent / "world.json", self.world)
        save((root / "prompts.json") if pilot else (parent / "baseline_prompts.json"), self.prompts)
        initial = deepcopy(self.world["public"]["initial"])
        state = deepcopy(initial)
        for event in ordered_events(self.world["public"]["events"]["high"], self.world["private"]["popular"], "pair_0"):
            before = deepcopy(state)
            state["users"][event["user"]] += " Observed consumption."
            state["items"][event["positive"]]["memory"] += " Observed by one user."
            state["completed"].append(event["event_id"])
            save(arm / (event["event_id"] + ".json"), {"event": event, "before": before, "after": deepcopy(state)})
        save(arm / "state.json", state)
        save(parent / "initial/state.json", initial)
        rankings = {direction: {q["query_id"]: list(q["candidates"]) for q in self.world["public"]["queries"]}
                    for direction in ("forward", "reverse")}
        save(arm / "rankings.json", rankings)
        save(parent / "initial/rankings.json", rankings)
        save(root / "protocol.json", {"api_protocol": API_PROTOCOL, "ranking_protocol": RANKING_PROTOCOL})
        return arm

    def test_reads_trial_and_pilot_layouts_without_mutation(self):
        for pilot in (False, True):
            with self.subTest(pilot=pilot), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
                arm = self.fixture(tmp, pilot)
                before = {str(p): p.read_bytes() for p in Path(tmp).rglob("*.json")}
                snapshot = load_baseline(arm, self.world, self.prompts)
                self.assertEqual(source_prompts(arm), self.prompts)
                self.assertEqual(len(snapshot["records"]), 4)
                self.assertEqual(snapshot["events"], [row["event"] for row in snapshot["records"]])
                self.assertEqual(snapshot["state"], snapshot["records"][-1]["after"])
                self.assertEqual(snapshot["contract"]["metadata"]["api_protocol"], API_PROTOCOL)
                self.assertEqual(snapshot["contract"]["metadata"]["model_parameters"], "unknown")
                self.assertEqual(before, {str(p): p.read_bytes() for p in Path(tmp).rglob("*.json")})

    def test_world_and_prompt_changes_cannot_reuse_the_baseline(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            arm = self.fixture(tmp)
            world = deepcopy(self.world)
            world["public"]["queries"][0]["candidates"].reverse()
            with self.assertRaisesRegex(ValueError, "world differs"):
                load_baseline(arm, world, self.prompts)
            with self.assertRaisesRegex(ValueError, "prompts differ"):
                load_baseline(arm, self.world, self.prompts | {"new_prompt": "changed"})

    def test_final_state_and_initial_state_must_match_committed_records(self):
        for component in ("final", "initial"):
            with self.subTest(component=component), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
                arm = self.fixture(tmp)
                path = arm / "state.json" if component == "final" else arm.parent / "initial/state.json"
                state = json.loads(path.read_text(encoding="utf-8"))
                state["users"][next(iter(state["users"]))] = "An unrelated state"
                save(path, state)
                with self.assertRaisesRegex(ValueError, "state"):
                    load_baseline(arm, self.world, self.prompts)

    def test_event_chain_and_full_event_sequence_are_required(self):
        for corruption in ("chain", "missing"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
                arm = self.fixture(tmp)
                if corruption == "chain":
                    path = arm / "train-0001.json"
                    record = json.loads(path.read_text(encoding="utf-8"))
                    record["before"]["users"][next(iter(record["before"]["users"]))] = "Broken read"
                    save(path, record)
                else:
                    path = arm / "state.json"
                    record = json.loads((arm / "train-0002.json").read_text(encoding="utf-8"))
                    save(path, record["after"])
                with self.assertRaises(ValueError):
                    load_baseline(arm, self.world, self.prompts)

    def test_normal_shared_memory_is_required(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            arm = self.fixture(tmp)
            save(arm / "memory_control.json", {"memory_policy": "no_sharing", "feedback_protocol": MEMORY_CONTROL_PROTOCOL})
            with self.assertRaisesRegex(ValueError, "shared-memory"):
                load_baseline(arm, self.world, self.prompts)

    def test_panels_and_query_coverage_are_checked_but_none_failures_are_retained(self):
        for corruption in ("missing_query", "duplicate_item", "recorded_failure"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
                arm = self.fixture(tmp)
                path = arm / "rankings.json"
                pair = json.loads(path.read_text(encoding="utf-8"))
                qid = next(iter(pair["forward"]))
                if corruption == "missing_query":
                    pair["forward"].pop(qid)
                elif corruption == "duplicate_item":
                    pair["forward"][qid][0] = pair["forward"][qid][1]
                else:
                    pair["forward"][qid] = None
                save(path, pair)
                if corruption == "recorded_failure":
                    self.assertIsNone(load_baseline(arm, self.world, self.prompts)["baseline_pair"][0][qid])
                else:
                    with self.assertRaises(ValueError):
                        load_baseline(arm, self.world, self.prompts)

    def test_recorded_protocol_or_model_mismatch_is_rejected(self):
        for bad in ({"api_protocol": "old_api"}, {"ranking_protocol": "old_ranking"},
                    {"model_parameters": MODEL_PARAMETERS | {"temperature": 1}}):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
                arm = self.fixture(tmp)
                save(arm.parent / "protocol.json", bad)
                with self.assertRaisesRegex(ValueError, "differs"):
                    load_baseline(arm, self.world, self.prompts)

    def test_model_config_and_actual_requests_are_used_when_recorded(self):
        for location in ("model_config", "calls"):
            with self.subTest(location=location), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
                arm = self.fixture(tmp, pilot=True)
                if location == "model_config":
                    save(Path(tmp) / "model_config.json", {"parameters": MODEL_PARAMETERS, "api_protocol": API_PROTOCOL})
                else:
                    row = {"status": "started", "api_protocol": API_PROTOCOL,
                           "context": {"world": "world", "pilot_arm": "original_update"},
                           "request": MODEL_PARAMETERS | {"messages": [{"content": "not included in contract"}]}}
                    (Path(tmp) / "calls.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
                contract = load_baseline(arm, self.world, self.prompts)["contract"]
                self.assertEqual(contract["metadata"]["model_parameters"], MODEL_PARAMETERS)
                self.assertNotIn("messages", json.dumps(contract))

    def test_partial_old_request_does_not_invent_parameters_and_cannot_hide_a_mismatch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            arm = self.fixture(tmp)
            path = arm.parent / "calls.jsonl"
            row = {"status": "request", "request": {"model": MODEL_PARAMETERS["model"]}}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            snapshot = load_baseline(arm, self.world, self.prompts)
            self.assertEqual(snapshot["contract"]["metadata"]["model_parameters"], "unknown")
            row["request"]["model"] = "a_different_model"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model parameter differs"):
                load_baseline(arm, self.world, self.prompts)

    def test_contract_identity_survives_copy_and_does_not_promote_unknown_metadata(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            arm = self.fixture(Path(tmp) / "first")
            events = ordered_events(self.world["public"]["events"]["high"], self.world["private"]["popular"], "pair_0")
            contract = make_baseline_contract(self.world, self.prompts, events,
                                             {"api_protocol": API_PROTOCOL, "ranking_protocol": RANKING_PROTOCOL,
                                              "model_parameters": MODEL_PARAMETERS})
            save(arm.parent / "baseline_contract.json", contract)
            save(arm / "memory_control.json", {"memory_policy": "original_update", "feedback_protocol": MEMORY_CONTROL_PROTOCOL})
            first = load_baseline(arm, self.world, self.prompts)
            copied = Path(tmp) / "second"
            shutil.copytree(arm.parent, copied)
            second = load_baseline(copied / "B0", self.world, self.prompts)
            self.assertEqual(first["contract"], contract)
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertEqual(second["contract"]["metadata"]["memory_control_protocol"], "unknown")
            changed = deepcopy(contract)
            changed["events_hash"] = "different"
            save(copied / "baseline_contract.json", changed)
            with self.assertRaisesRegex(ValueError, "contract differs"):
                load_baseline(copied / "B0", self.world, self.prompts)

    def test_original_prompt_wrapper_and_ambiguous_records(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            arm = self.fixture(tmp)
            (arm.parent / "baseline_prompts.json").unlink()
            save(arm.parent / "prompts.json", {"original": self.prompts, "repair": {"changed": "prompt"}})
            self.assertEqual(source_prompts(arm), self.prompts)
            save(arm.parent / "baseline_prompts.json", self.prompts | {"different": "value"})
            with self.assertRaisesRegex(ValueError, "disagree"):
                source_prompts(arm)

    def test_offline_baseline_requires_an_explicit_offline_adapter_expectation(self):
        from rsi_researcher.runtime.offline import OfflineResearchModel
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            arm = self.fixture(tmp)
            metadata = OfflineResearchModel().metadata
            save(arm.parent / "protocol.json", metadata)
            with self.assertRaisesRegex(ValueError, "api_protocol differs"):
                load_baseline(arm, self.world, self.prompts)
            snapshot = load_baseline(arm, self.world, self.prompts, metadata=metadata)
            self.assertEqual(snapshot["contract"]["metadata"], metadata)


if __name__ == "__main__":
    unittest.main()
