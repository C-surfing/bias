"""KuaiRand read-only transfer and non-identifiability regression tests."""
from pathlib import Path
import tempfile
import unittest
import rsi_researcher.research.unified as u
from rsi_researcher.research.planner_adapters import load_real_situations
from rsi_researcher.research.planner_integration import minimal_proposal, execute_read_only
from rsi_researcher.research.kuairand_transfer import run


class ExposureTransferTests(unittest.TestCase):
    def test_exposure_audit_reuses_existing_gate(self):
        context = next(c for c in load_real_situations()
                       if c.situation_id == "logging-support")
        result = execute_read_only(context, minimal_proposal(context, "audit_common_support"))
        self.assertEqual(result["status"], "CPU_COMPLETED")
        self.assertEqual(result["result"]["common_item_count"], 201)

    def test_missing_identification_blocks_replication(self):
        context = next(c for c in load_real_situations()
                       if c.situation_id == "logging-support")
        blocked = execute_read_only(
            context, minimal_proposal(context, "run_independent_replication"))
        self.assertEqual(blocked["status"], "BLOCKED")

    def test_transfer_writes_development_only_memory(self):
        with tempfile.TemporaryDirectory(dir=u.HERE / "runs") as parent:
            result = run(Path(parent) / "exposure")
            self.assertEqual(result["status"], "EXPOSURE_IDENTIFICATION_NOT_YET_POSSIBLE")
            self.assertFalse(result["independent_confirmation"])
            self.assertEqual(result["new_llm_requests"], 0)
            self.assertEqual(result["gpu_runs"], 0)


if __name__ == "__main__":
    unittest.main()
