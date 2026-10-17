"""Fetch the two fixed official datasets and preserve their source provenance."""

from datetime import datetime
from hashlib import md5, sha256
import http.client
import json
from pathlib import Path, PurePosixPath
import shutil
import ssl
import stat
import tarfile
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

import certifi


DATASETS = {
    "coat": {
        "dataset": "Coat Shopping",
        "official_page": "https://www.cs.cornell.edu/~schnabts/mnar/",
        "download_url": "https://www.cs.cornell.edu/~schnabts/mnar/coat.zip",
        "archive_name": "coat.zip",
        "archive_bytes": 537499,
        "sha256": "6073d0b515ed1f6e830e4fead66dc76ad7991a7553eaa58e228b234a9d19daed",
        "license": "CC BY-NC 4.0 as stated on author page",
    },
    "kuairand-pure": {
        "dataset": "KuaiRand-Pure",
        "official_page": "https://kuairand.com/",
        "download_url": "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz",
        "archive_name": "KuaiRand-Pure.tar.gz",
        "archive_bytes": 47432272,
        "md5": "0820331067a3784d9691136f772b35a7",
        "license": "CC BY 4.0 (Zenodo dataset record)",
    },
}

CHUNK_BYTES = 1024 * 1024
MAX_ATTEMPTS = 3


def _hashes(path):
    sha, legacy = sha256(), md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            sha.update(chunk)
            legacy.update(chunk)
    return {"sha256": sha.hexdigest(), "md5": legacy.hexdigest()}


def _verify_archive(path, spec):
    hashes = _hashes(path)
    if path.stat().st_size != spec["archive_bytes"]:
        raise ValueError("Dataset archive size does not match the official source")
    for algorithm in ("sha256", "md5"):
        if algorithm in spec and hashes[algorithm] != spec[algorithm]:
            raise ValueError(f"Dataset archive failed its official {algorithm} checksum")
    return hashes


def _download(path, spec):
    """Append complete verified HTTP ranges; leave successful ranges on failure."""
    partial = path.with_name(path.name + ".partial")
    marker = path.with_name(path.name + ".partial.json")
    identity = {key: spec[key] for key in ("download_url", "archive_bytes")}
    identity.update({key: spec[key] for key in ("sha256", "md5") if key in spec})
    if partial.exists():
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise FileExistsError("An unrecognized partial archive exists; use another output directory")
    elif marker.exists():
        raise FileExistsError("A partial marker without its archive exists; use another output directory")
    else:
        marker.write_text(json.dumps(identity, indent=2), encoding="utf-8")
        partial.touch(exist_ok=False)
    total = spec["archive_bytes"]
    if partial.stat().st_size > total:
        raise ValueError("Partial archive is larger than the official archive")
    context = ssl.create_default_context(cafile=certifi.where())
    while (start := partial.stat().st_size) < total:
        end = min(start + CHUNK_BYTES, total) - 1
        request = Request(spec["download_url"], headers={
            "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
            "User-Agent": "rsi-researcher/official-dataset-fetch",
        })
        for attempt in range(MAX_ATTEMPTS):
            try:
                with urlopen(request, timeout=30, context=context) as response:
                    status = response.status
                    if status == 206:
                        expected = f"bytes {start}-{end}/{total}"
                        if response.headers.get("Content-Range") != expected:
                            raise ValueError("HTTP Content-Range does not match the requested dataset chunk")
                        data = response.read(end - start + 2)
                        if len(data) != end - start + 1:
                            raise http.client.IncompleteRead(data, end - start + 1)
                        full = False
                    elif status == 200:
                        # Some official hosts ignore Range. Accept only a complete response.
                        data = response.read(total + 1)
                        if len(data) != total:
                            raise http.client.IncompleteRead(data, total)
                        full = True
                    else:
                        raise ValueError(f"Unexpected dataset HTTP status: {status}")
                break
            except (URLError, TimeoutError, http.client.IncompleteRead, ConnectionError) as error:
                if attempt + 1 == MAX_ATTEMPTS:
                    raise RuntimeError("Official download interrupted; rerun to resume saved chunks") from error
        with partial.open("wb" if full else "ab") as handle:
            handle.write(data)
    _verify_archive(partial, spec)
    partial.rename(path)
    marker.unlink()


def _destination(root, name):
    relative = PurePosixPath(name)
    if (not name or "\\" in name or relative.is_absolute() or ".." in relative.parts
            or any(":" in part for part in relative.parts)):
        raise ValueError(f"Unsafe dataset archive path: {name!r}")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Dataset archive path escapes the output directory: {name!r}")
    for parent in (target, *target.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError(f"Dataset extraction target is a symbolic link: {name!r}")
    return target


def _extract(archive, root):
    """Validate the whole member list before writing, and never replace unknown files."""
    container = zipfile.ZipFile(archive) if archive.suffix == ".zip" else tarfile.open(archive, "r:gz")
    with container:
        members = container.infolist() if isinstance(container, zipfile.ZipFile) else container.getmembers()
        regular = []
        destinations = set()
        for member in members:
            name = member.filename if isinstance(member, zipfile.ZipInfo) else member.name
            target = _destination(root, name)
            if isinstance(member, zipfile.ZipInfo):
                kind = stat.S_IFMT(member.external_attr >> 16)
                directory = member.is_dir()
                valid = kind in (0, stat.S_IFREG, stat.S_IFDIR) and (directory or kind != stat.S_IFDIR)
            else:
                directory, valid = member.isdir(), member.isfile() or member.isdir()
            if not valid:
                raise ValueError(f"Dataset archive contains a link or special file: {name!r}")
            if directory:
                continue
            if target in destinations:
                raise ValueError(f"Dataset archive contains duplicate files: {name!r}")
            destinations.add(target)
            regular.append((member, target, name))
        # Check conflicts before extracting anything. Existing identical files can be reused.
        for member, target, name in regular:
            if not target.exists():
                continue
            if not target.is_file():
                raise FileExistsError(f"Dataset extraction would overwrite an existing path: {name}")
            with (container.open(member) if isinstance(container, zipfile.ZipFile)
                  else container.extractfile(member)) as handle:
                expected = sha256()
                for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
                    expected.update(chunk)
            if _hashes(target)["sha256"] != expected.hexdigest():
                raise FileExistsError(f"Dataset extraction would overwrite a different file: {name}")
        files = {}
        for member, target, name in regular:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with (container.open(member) if isinstance(container, zipfile.ZipFile)
                      else container.extractfile(member)) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, CHUNK_BYTES)
            files[name] = {"bytes": target.stat().st_size, "sha256": _hashes(target)["sha256"]}
        return files


def fetch_dataset(dataset, output):
    """Return source.json for ``coat`` or ``kuairand-pure`` in its own directory.

    Verified local archives are reused without a network request. A previous
    source manifest is returned unchanged after its archive and extracted-file
    hashes pass; partial official downloads can be resumed by repeating this call.
    """
    if dataset not in DATASETS:
        raise ValueError("dataset must be 'coat' or 'kuairand-pure'")
    spec, root = DATASETS[dataset], Path(output)
    root.mkdir(parents=True, exist_ok=True)
    archive, source_path = root / spec["archive_name"], root / "source.json"
    previous = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else None
    if previous is not None and (not isinstance(previous, dict)
                                or previous.get("dataset") != spec["dataset"]
                                or previous.get("download_url") != spec["download_url"]):
        raise FileExistsError("source.json belongs to another source; use another output directory")
    if archive.exists():
        hashes = _verify_archive(archive, spec)
    else:
        _download(archive, spec)
        hashes = _verify_archive(archive, spec)
    if previous is not None:
        if previous.get("sha256") != hashes["sha256"]:
            raise ValueError("source.json does not match the local official archive")
        files = previous.get("files", {})
        if files and all(_destination(root, name).is_file() for name in files):
            for name, expected in files.items():
                digest = expected if isinstance(expected, str) else expected["sha256"]
                if _hashes(_destination(root, name))["sha256"] != digest:
                    raise ValueError(f"Extracted dataset file failed its recorded checksum: {name}")
            return previous
    files = _extract(archive, root)
    manifest = {key: value for key, value in spec.items() if key != "archive_name"}
    manifest.update({"sha256": hashes["sha256"], "files": files,
                     "download_date": datetime.now().date().isoformat()})
    source_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
