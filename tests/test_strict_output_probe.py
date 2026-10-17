import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from rsi_researcher.runtime.client import InvalidOutput, logged_response_content, usage_summary
from rsi_researcher.diagnostics.strict_output import StrictDeepSeek


class StrictOutputTests(unittest.TestCase):
    def test_explicit_schema_and_one_submission_use_existing_budget_and_logging(self):
        schema = {"type": "object", "properties": {"memory": {"type": "string", "pattern": "\\S"}},
                  "required": ["memory"], "additionalProperties": False}
        body = {"choices": [{"message": {"content": None, "tool_calls": [{"type": "function", "function": {
                "name": "submit_result", "arguments": '{"memory":"kept"}'}}]}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}):
            model = StrictDeepSeek(tmp, 3)
            model.scope = {"workflow_arm": "B0"}
            with patch("urllib.request.urlopen") as network:
                network.return_value.__enter__.return_value.read.return_value = json.dumps(body).encode()
                self.assertEqual(model.ask("prompt", schema, {}), {"memory": "kept"})
            request = network.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual(request.full_url, "https://api.deepseek.com/beta/chat/completions")
            self.assertTrue(payload["tools"][0]["function"]["strict"])
            self.assertEqual(payload["tools"][0]["function"]["parameters"], schema)
            self.assertNotIn("response_format", payload)
            self.assertEqual(usage_summary(tmp)["prompt_tokens"], 10)
            started = json.loads((Path(tmp) / "calls.jsonl").read_text().splitlines()[0])
            self.assertEqual(started["context"]["workflow_arm"], "B0")
            self.assertNotIn("workflow_arm", json.dumps(payload))

    def test_ambiguous_or_unknown_submissions_are_rejected(self):
        model = object.__new__(StrictDeepSeek)
        for calls in ([], [{"type": "function", "function": {"name": "other", "arguments": "{}"}}], [{}, {}]):
            with self.assertRaises(InvalidOutput):
                model.response_content({"message": {"tool_calls": calls}})

    def test_both_log_formats_remain_readable(self):
        text = '{"ranking":["a","b"]}'
        self.assertEqual(logged_response_content({"content": text}), text)
        self.assertEqual(logged_response_content({"content": None, "tool_calls": [{"type": "function",
                         "function": {"name": "submit_result", "arguments": text}}]}), text)

    def test_missing_field_is_retried_within_same_request_budget(self):
        schema = {"type": "object", "properties": {"first_memory": {"type": "string", "pattern": "\\S"},
                   "second_memory": {"type": "string", "pattern": "\\S"}},
                  "required": ["first_memory", "second_memory"], "additionalProperties": False}
        def packet(result):
            return json.dumps({"choices": [{"message": {"content": None, "tool_calls": [{"type": "function",
                "function": {"name": "submit_result", "arguments": json.dumps(result)}}]}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
        good = {"first_memory": "first", "second_memory": "second"}
        for results, succeeds in (([{"first_memory": "first"}, good], True), ([{"first_memory": "first"}] * 3, False)):
            with self.subTest(succeeds=succeeds), tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}):
                model = StrictDeepSeek(tmp, 3)
                with patch("urllib.request.urlopen") as network, patch("time.sleep"):
                    network.return_value.__enter__.return_value.read.side_effect = [packet(r) for r in results]
                    if succeeds:
                        self.assertEqual(model.ask("prompt", schema, {}), good)
                    else:
                        with self.assertRaises(InvalidOutput):model.ask("prompt", schema, {})
                self.assertEqual(network.call_count, len(results))
                self.assertEqual(usage_summary(tmp)["prompt_tokens"], 10 * len(results))
                second_payload = json.loads(network.call_args_list[1].args[0].data)
                self.assertIn("Required top-level fields: first_memory, second_memory", second_payload["messages"][0]["content"])

    def test_local_validation_rejects_wrong_types_extra_fields_and_blank_text(self):
        schema = {"type": "object", "properties": {"memory": {"type": "string", "pattern": "\\S"}},
                  "required": ["memory"], "additionalProperties": False}
        model = object.__new__(StrictDeepSeek)
        for value in ({}, {"memory": " "}, {"memory": 1}, {"memory": "valid", "extra": "x"}):
            with self.assertRaises(InvalidOutput):model.validate_result(value, schema)

    def test_retry_identifies_forbidden_strength_and_lists_allowed_values(self):
        schema = {"type": "object", "properties": {"next_strength": {"type": "number", "enum": [0, .05, .1]}},
                  "required": ["next_strength"], "additionalProperties": False}
        def packet(strength):
            return json.dumps({"choices": [{"message": {"content": None, "tool_calls": [{"type": "function",
                "function": {"name": "submit_result", "arguments": json.dumps({"next_strength": strength})}}]},
                "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}):
            model = StrictDeepSeek(tmp, 2)
            with patch("urllib.request.urlopen") as network, patch("time.sleep"):
                network.return_value.__enter__.return_value.read.side_effect = [packet(.02), packet(.05)]
                self.assertEqual(model.ask("choose the next experiment", schema, {}), {"next_strength": .05})
            retry = json.loads(network.call_args_list[1].args[0].data)
            self.assertIn("response field next_strength", retry["messages"][0]["content"])
            self.assertIn("allowed values: [0, 0.05, 0.1]", retry["messages"][0]["content"])
            self.assertEqual(usage_summary(tmp)["requests"], 2)


if __name__ == "__main__":
    unittest.main()
