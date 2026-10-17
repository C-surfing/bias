"""Integration checks for the package entry point and frozen worker imports."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsi_researcher.paths import ROOT
from rsi_researcher.runtime import source_snapshot


class ProjectStructureTests(unittest.TestCase):
    def make_project(self, directory):
        project = directory / "project"
        package = project / "rsi_researcher"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "worker.py").write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "print(json.dumps({'source':str(Path(__file__).resolve()),'args':sys.argv[1:]}))\n",
            encoding="utf-8")
        (project / "requirements.txt").write_text("", encoding="utf-8")
        return project

    def test_frozen_worker_wins_over_a_different_current_directory_package(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            directory = Path(temporary)
            project = self.make_project(directory)
            run = directory / "run with spaces"
            with patch.object(source_snapshot, "ROOT", project):
                manifest = source_snapshot.snapshot_package(run)
                self.assertEqual(source_snapshot.verify_snapshot(run), manifest)
            shadow = directory / "different checkout"
            (shadow / "rsi_researcher").mkdir(parents=True)
            (shadow / "rsi_researcher/__init__.py").write_text(
                "raise RuntimeError('current checkout was imported instead of frozen worker')\n", encoding="utf-8")
            arguments = ["two words", "中文参数", "quote'and\"double"]
            result = subprocess.run(source_snapshot.module_command(run, "rsi_researcher.worker", arguments),
                                    cwd=shadow, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            reply = json.loads(result.stdout)
            self.assertEqual(Path(reply["source"]), run / "code/rsi_researcher/worker.py")
            self.assertEqual(reply["args"], arguments)

    def test_snapshot_verification_checks_saved_and_current_implementation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            directory = Path(temporary)
            project = self.make_project(directory)
            run = directory / "run"
            with patch.object(source_snapshot, "ROOT", project):
                source_snapshot.snapshot_package(run)
                frozen = run / "code/rsi_researcher/worker.py"
                original = frozen.read_bytes()
                frozen.write_bytes(original + b"\n# altered snapshot\n")
                with self.assertRaisesRegex(ValueError, "saved implementation changed"):
                    source_snapshot.verify_snapshot(run)
                frozen.write_bytes(original)
                (project / "rsi_researcher/worker.py").write_bytes(original + b"\n# altered checkout\n")
                with self.assertRaisesRegex(ValueError, "current implementation differs"):
                    source_snapshot.verify_snapshot(run)
                with self.assertRaises(FileExistsError):
                    source_snapshot.snapshot_package(run)
                self.assertEqual(frozen.read_bytes(), original)

    def test_public_offline_entry_point_works_without_an_api_key(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary) / "offline demo"
            environment = {name: value for name, value in os.environ.items() if name != "DEEPSEEK_API_KEY"}
            environment["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run([sys.executable, "-m", "rsi_researcher", "demo", "--output", str(output)],
                                    cwd=ROOT, env=environment, capture_output=True, text=True,
                                    encoding="utf-8", timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            marker = json.loads((output / "offline_demo.json").read_text(encoding="utf-8"))
            self.assertEqual(marker["paid_api_calls"], 0)
            self.assertEqual(marker["network_calls"], 0)
            archive = json.loads((output / "archive.json").read_text(encoding="utf-8"))
            self.assertEqual(archive["execution_status"], "completed")
            self.assertTrue(archive["trials"])


if __name__ == "__main__":
    unittest.main()
