import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from rsi_researcher.runtime.client import BudgetExceeded, DeepSeek
from rsi_researcher.runtime.limits import RunLimits


class LimitsTests(unittest.TestCase):
    def test_no_outbound_request_when_reservation_does_not_fit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}):
            limits = RunLimits(tmp, 5, 1, 60)
            model = DeepSeek(tmp, 5, limits)
            with patch("urllib.request.urlopen") as network, self.assertRaises(BudgetExceeded):
                model.ask("hello", "{}", {})
            network.assert_not_called()
            self.assertEqual(model.requests, 0)

    def test_unknown_usage_and_retries_are_charged_and_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}):
            limits = RunLimits(tmp, 5, 5000, 60)
            model = DeepSeek(tmp, 5, limits)
            with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("test")) as network, patch("time.sleep"):
                with self.assertRaises(BudgetExceeded):
                    model.ask("hello", "{}", {})
            self.assertEqual(network.call_count, 1)
            self.assertEqual(RunLimits(tmp, 5, 5000, 60).charged_tokens, limits.charged_tokens)
            with self.assertRaises(ValueError):
                RunLimits(tmp, 5, 10000, 60)
            with self.assertRaises(ValueError):
                DeepSeek(tmp, 5)

    def test_reported_usage_replaces_reservation_and_restores_from_log(self):
        body = json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 10}}).encode()
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}):
            limits = RunLimits(tmp, 5, 5000, 60)
            with patch("urllib.request.urlopen") as network:
                network.return_value.__enter__.return_value.read.return_value = body
                DeepSeek(tmp, 5, limits).ask("hello", "{}", {})
            self.assertEqual(limits.charged_tokens, 110)
            self.assertEqual(RunLimits(tmp, 5, 5000, 60).charged_tokens, 110)

    def test_deadline_does_not_restart_and_late_result_cannot_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("time.time", return_value=100), patch("time.monotonic", return_value=50):
                limits = RunLimits(tmp, 5, 5000, 10)
                reservation = limits.reserve({"max_tokens": 10})
            with patch("time.time", return_value=111), patch("time.monotonic", return_value=61):
                with self.assertRaises(BudgetExceeded):
                    limits.settle(reservation, {"total_tokens": 10})
                resumed = RunLimits(tmp, 5, 5000, 10)
                with self.assertRaises(BudgetExceeded):
                    resumed.remaining_seconds()


if __name__ == "__main__":
    unittest.main()
