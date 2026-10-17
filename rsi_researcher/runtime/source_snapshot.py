"""Freeze the runnable package so workers use the recorded implementation."""

import hashlib
import json
from pathlib import Path
import sys

from rsi_researcher.paths import ROOT


def _sources():
    package = ROOT / "rsi_researcher"
    files = {path.relative_to(ROOT).as_posix(): path for path in sorted(package.rglob("*.py"))}
    files["requirements.txt"] = ROOT / "requirements.txt"
    return files


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def snapshot_package(run_dir):
    """Write a complete package under code/ and return its relative-path hashes.

    Existing snapshots are never replaced. Inputs and results belong to the run
    directory; this snapshot contains the implementation and its requirements.
    """
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "source_manifest.json").exists():
        raise FileExistsError("run already has a source manifest")
    code = run_dir / "code"
    code.mkdir(exist_ok=False)
    manifest = {}
    for relative, source in _sources().items():
        content = source.read_bytes()
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest[relative] = _digest(content)
    (run_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_snapshot(run_dir):
    """Check both the saved code and the current implementation against hashes."""
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "source_manifest.json").read_text(encoding="utf-8"))
    sources = _sources()
    saved_names = {path.relative_to(run_dir / "code").as_posix()
                   for path in (run_dir / "code" / "rsi_researcher").rglob("*.py")}
    saved_names.add("requirements.txt")
    if set(manifest) != set(sources) or set(manifest) != saved_names:
        raise ValueError("snapshot package members differ from the recorded implementation")
    for relative, expected in manifest.items():
        if _digest((run_dir / "code" / relative).read_bytes()) != expected:
            raise ValueError(f"saved implementation changed: {relative}")
        if _digest(sources[relative].read_bytes()) != expected:
            raise ValueError(f"current implementation differs from the frozen source: {relative}")
    return manifest


def module_command(run_dir, module, args=()):
    """Build a fresh-process command which imports the package from code/ first."""
    if not module.startswith("rsi_researcher.") and module != "rsi_researcher":
        raise ValueError("worker module must belong to rsi_researcher")
    code = str((Path(run_dir).resolve() / "code"))
    argv = [module, *(str(value) for value in args)]
    bootstrap = ("import runpy, sys; "
                 f"sys.path.insert(0, {code!r}); "
                 f"sys.argv = {argv!r}; "
                 f"runpy.run_module({module!r}, run_name='__main__')")
    return [sys.executable, "-c", bootstrap]
