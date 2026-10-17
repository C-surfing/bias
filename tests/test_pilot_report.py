import unittest

from rsi_researcher.evaluation.pilot_report import request_diagnostics, request_markdown, summarize


class PilotReportTests(unittest.TestCase):
    def test_seed_pairing_equal_strata_and_incomplete_batch(self):
        protocol = {"seeds": [1, 2, 3], "strata": ["a", "b"]}
        rows = [{"seed": seed, "stratum": layer, "utility_gain": value, "bias_change": 0}
                for seed, layer, value in ((1, "a", 0), (1, "b", 1), (2, "a", 1))]
        result = summarize(protocol, rows, None)
        self.assertFalse(result["complete"])
        self.assertEqual(result["independent_seed_count"], 1)
        self.assertEqual(result["utility_gain_mean"], 0.5)
        self.assertEqual(len(result["missing_worlds"]), 3)
        with self.assertRaises(ValueError):
            summarize(protocol, rows + rows[:1], "completed")

    def test_all_rows_need_successful_supervisor_exit(self):
        protocol = {"seeds": [1], "strata": ["a"]}
        rows = [{"seed": 1, "stratum": "a", "utility_gain": 0, "bias_change": 0}]
        self.assertFalse(summarize(protocol, rows, "timed_out")["complete"])
        self.assertTrue(summarize(protocol, rows, "completed")["complete"])

    def test_request_diagnostics_include_failed_and_unknown_usage(self):
        records = [
            {"status": "started", "call_id": 1, "attempt": 1, "context": {"stage": "select"}, "time_utc": "2026-10-13T00:00:00+00:00"},
            {"status": "received", "call_id": 1, "seconds": 2, "usage": {"prompt_tokens": 10, "completion_tokens": 1}},
            {"status": "started", "call_id": 2, "attempt": 2, "context": {"stage": "select"}, "time_utc": "2026-10-13T00:00:04+00:00"},
            {"status": "failed", "call_id": 2, "seconds": 3},
            {"status": "started", "call_id": 3, "attempt": 1, "context": {"stage": "rank"}, "time_utc": "2026-10-13T00:00:08+00:00"},
            {"status": "received", "call_id": 3, "seconds": 6, "usage": None},
            {"status": "ranking_correction_succeeded"},
            {"status": "invalid_response", "call_id": 1, "reason": "schema_violation"},
            {"status": "invalid_response", "call_id": 3, "reason": "empty_content"},
        ]
        result = request_diagnostics(records)
        self.assertEqual((result["requests"], result["responses"], result["transport_failures"]), (3, 2, 1))
        self.assertEqual(result["requests_with_unknown_usage"], 2)
        self.assertEqual(result["retry_attempts"], 1)
        self.assertEqual(result["schema_violations"], 1)
        self.assertEqual(result["empty_responses"], 1)
        self.assertIn("本地字段校验拒绝 1 份响应", request_markdown(result))
        self.assertEqual(result["response_median_seconds"], 4)
        self.assertEqual(result["response_p95_seconds"], 6)
        self.assertEqual(result["observed_request_window_seconds"], 14)
        self.assertEqual(result["summed_recorded_request_seconds"], 11)
        self.assertIn("充值等待", request_markdown(result))
        self.assertEqual(result["stages"]["select"]["requests"], 2)
        with self.assertRaises(ValueError):
            request_diagnostics(records + records[:1])


if __name__ == "__main__":
    unittest.main()
