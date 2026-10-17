"""Download the official checkpoint at a recorded immutable revision."""
import argparse
import json
import hashlib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--base', default='/data/cjh/rsi')
p.add_argument('--source', choices=['huggingface', 'modelscope'], default='huggingface')
args = p.parse_args()
base = Path(args.base).resolve()
repo = 'Qwen/Qwen3-8B'
manifest = base / 'models/qwen3-8b-manifest.json'
previous = json.loads(manifest.read_text()) if manifest.exists() else {}
if previous and previous['source'] != args.source:
    raise ValueError('Do not silently change the source of an existing model directory')
if args.source == 'huggingface':
    from huggingface_hub import HfApi, snapshot_download
    revision = previous.get('revision') or HfApi().model_info(repo).sha
else:
    from modelscope import snapshot_download
    revision = previous.get('revision', 'master')
directory = base / 'models/Qwen3-8B'
manifest.parent.mkdir(parents=True, exist_ok=True)
# Save the selected version even when a download is interrupted.
manifest.write_text(json.dumps({'repository': repo, 'source': args.source, 'revision': revision,
    'directory': str(directory), 'status': 'downloading'}, indent=2) + '\n')
if args.source == 'huggingface':
    snapshot_download(repo, revision=revision, local_dir=directory,
                      allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', 'LICENSE', 'README.md'],
                      max_workers=4)
else:
    snapshot_download(repo, revision=revision, local_dir=str(directory))
files = {}
for path in sorted(directory.rglob('*')):
    if not path.is_file() or any(part.startswith('.') for part in path.relative_to(directory).parts):
        continue
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    files[path.relative_to(directory).as_posix()] = {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}
manifest.write_text(json.dumps({'repository': repo, 'source': args.source, 'revision': revision,
    'directory': str(directory), 'status': 'downloaded', 'files': files,
    'revision_note': 'ModelScope master is mutable; recorded file hashes identify the actual downloaded snapshot.'}, indent=2) + '\n')
print('Official model downloaded; revision:', revision)
