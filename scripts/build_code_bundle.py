"""Build a complete, structured source ZIP without historical API logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("README.md", "pyproject.toml", "requirements.txt")
SOURCE_TREES = {
    "rsi_researcher": {".py"},
    "tests": {".py"},
    "scripts": {".py"},
    "docs": {".md", ".csv", ".json"},
    "configs": {".json", ".yaml", ".yml", ".md", ".csv"},
    "deployment": {".sh", ".py", ".md", ".json", ".ps1"},
}
DATA_FILES = ("README.md", "author_prompts.json", "CDs.item", "CDs-100-user-dense.train.inter",
              "CDs-100-user-dense.valid.inter", "CDs-100-user-dense.random")
DENIED_PARTS = {"__pycache__", ".git", ".venv", "runs", "outputs"}
DENIED_LOGS = {"calls.jsonl", "worker.log", "usage.json", "balance.json", "balance_after.json"}
OFFLINE_SUFFIXES = {".json", ".md", ".txt", ".csv"}
SECRET_PATTERN = re.compile(rb"(?:sk-[A-Za-z0-9_-]{20,}|Bearer[ \t]+[A-Za-z0-9._-]{24,})")


def checked_bytes(path):
    """Inspect selected deliverables only; never read environment or credential files."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Not a regular source file: {path.name}")
    content = path.read_bytes()
    if SECRET_PATTERN.search(content):
        raise ValueError(f"Possible literal credential in selected file: {path.name}")
    return content


def collect_sources(root, offline_dirs=()):
    root = Path(root).resolve()
    selected = {}

    def add(source, name=None):
        relative = name or source.relative_to(root).as_posix()
        if relative in selected:
            raise ValueError(f"Duplicate archive member: {relative}")
        selected[relative] = checked_bytes(source)

    for name in ROOT_FILES:
        add(root / name)
    for name, suffixes in SOURCE_TREES.items():
        directory = root / name
        if not directory.is_dir():
            raise ValueError(f"Missing project directory: {name}")
        for source in sorted(directory.rglob("*")):
            relative = source.relative_to(directory)
            if any(part in DENIED_PARTS or part.startswith(".") for part in relative.parts):
                continue
            if source.is_file() and source.suffix.lower() in suffixes:
                add(source)
    for name in DATA_FILES:
        add(root / "data" / name)
    # Keep upstream attribution without shipping its complete checkout.
    for name in ("LICENSE", "README.md"):
        candidates = (root / "sources" / ("AgentCF-" + name),
                      root / "vendor" / "AgentCF-reference" / name)
        source = next((candidate for candidate in candidates if candidate.exists()), None)
        if source:
            add(source, "sources/AgentCF-" + name)

    for index, directory in enumerate(offline_dirs, 1):
        directory = Path(directory).resolve()
        if not directory.is_relative_to(root) or directory.is_symlink():
            raise ValueError("Offline demonstrations must be regular directories inside the project")
        marker = json.loads(checked_bytes(directory / "offline_demo.json").decode("utf-8"))
        if any(marker.get(key) != value for key, value in
               (("kind", "offline_test_double"), ("paid_api_calls", 0), ("network_calls", 0))):
            raise ValueError("Demonstration is not explicitly marked as offline with zero API/network calls")
        for source in sorted(directory.rglob("*")):
            if source.is_symlink():
                raise ValueError("Offline demonstrations cannot contain symbolic links")
            if not source.is_file():
                continue
            if source.name in DENIED_LOGS or source.name.startswith(".env"):
                raise ValueError(f"API log or credential file found in offline demonstration: {source.name}")
            if source.suffix.lower() in OFFLINE_SUFFIXES:
                add(source, f"examples/offline-{index}/" + source.relative_to(directory).as_posix())
    return selected


def verify_bundle(path):
    """Check ZIP integrity, its member set, and every recorded SHA256."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
                name.startswith(("/", "\\")) or ".." in PurePosixPath(name).parts for name in names):
            raise ValueError("Duplicate or unsafe ZIP member")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP member: {bad}")
        manifest = json.loads(archive.read("source_manifest.json"))
        if set(names) != set(manifest["files"]) | {"source_manifest.json"}:
            raise ValueError("ZIP members do not match the source manifest")
        for name, expected in manifest["files"].items():
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != expected["sha256"] or len(content) != expected["bytes"]:
                raise ValueError(f"Source hash mismatch: {name}")
    return {"status": "verified", "files": len(names), "bytes": Path(path).stat().st_size,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def build_bundle(root=ROOT, output_dir=None, offline_dirs=(), deployment_only=False):
    root = Path(root).resolve()
    directory = Path(output_dir or root / "outputs").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Output directory must stay inside the project")
    if deployment_only:
        sources = {"deployment/qwen3/" + path.name: checked_bytes(path)
                   for path in sorted((root / "deployment/qwen3").iterdir()) if path.is_file()}
        for name in ("Qwen3-8B部署与后端替换.md", "Qwen服务器部署实际记录_2026-10-15.json"):
            sources["docs/" + name] = checked_bytes(root / "docs" / name)
        sources["project/requirements.txt"] = checked_bytes(root / "requirements.txt")
        sources["configs/qwen_backend_comparison.json"] = checked_bytes(root / "configs/qwen_backend_comparison.json")
    else:
        sources = collect_sources(root, offline_dirs)
    sources["交付说明.md"] = ("# 推荐系统研究员：结构化代码交付\n\n"
        "先读 README.md，再读 docs/代码导读.md。业务源码放在 rsi_researcher/，"
        "测试在 tests/，说明在 docs/，实验配置在 configs/，打包工具在 scripts/。\n\n"
        "解压后在包根目录运行 `python -m pip install -r requirements.txt`，"
        "再运行 `python -m unittest discover -s tests -t .`。"
        "用 `python -m rsi_researcher demo --output outputs/my-offline-demo` 查看离线研究过程。"
        "演示使用固定回复，不读取密钥，不代表真实研究能力或去偏效果。\n\n"
        "包内不含历史付费调用、密钥、完整 vendor 仓库和论文合集 ZIP。"
        "真实接口命令会产生模型费用，见代码导读；本交付不会自动启动它。\n\n"
        "source_manifest.json 记录每个交付文件的字节数与 SHA256。"
        "可运行 `python scripts/build_code_bundle.py --verify <本包路径>` 复查完整性。\n\n"
        "AgentCF 作者来源及数据限制见 data/README.md 和 sources/；"
        "固定作者版本为 71ff3b5bce69432556884c49fe7f19528b6756ec。"
        "sources/AgentCF-README.md 是上游原始说明，其中命令不等于本工作流入口。\n").encode("utf-8")
    if deployment_only:
        sources["交付说明.md"] = ("# Qwen3-8B 部署工具包\n\n先读 docs/Qwen3-8B部署与后端替换.md。"
            "本包建立独立环境、下载官方权重、启动/停止单卡服务并作独立接口检查。"
            "完整推荐系统研究代码在第二份源码包中，解压到服务器的 project 目录。"
            "实际部署状态见 docs/Qwen服务器部署实际记录_2026-10-15.json；脚本存在不表示部署已经完成。"
            "本包不含密码、模型权重或其他人的环境。source_manifest.json 记录文件校验值。\n").encode("utf-8")
    manifest = {"format": "recommendation_researcher_source_v2",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                          for name, content in sorted(sources.items())}}
    sources["source_manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    directory.mkdir(parents=True, exist_ok=True)
    filename = ("qwen3-8b-deployment-kit-" if deployment_only else "recommendation-researcher-structured-") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip"
    path = directory / filename
    created = False
    try:
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            created = True
            for name, content in sorted(sources.items()):
                archive.writestr(name, content)
        result = verify_bundle(path)
    except Exception:
        # Only the new, failed bundle can be removed; existing bundles stay untouched.
        if created and path.exists():
            path.unlink()
        raise
    return {"path": str(path), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Output directory inside the project; defaults to outputs")
    parser.add_argument("--offline-demo", action="append", default=[], help="Explicitly marked offline demonstration directory")
    parser.add_argument("--verify", help="Verify an existing bundle without creating another ZIP")
    parser.add_argument("--deployment-only", action="store_true", help="Build the separate Qwen deployment kit")
    args = parser.parse_args()
    result = verify_bundle(args.verify) if args.verify else build_bundle(output_dir=args.output_dir, offline_dirs=args.offline_demo, deployment_only=args.deployment_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
