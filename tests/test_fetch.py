"""Official fetching is exercised with small archives and simulated HTTP only."""

from hashlib import md5, sha256
from io import BytesIO
import json
from pathlib import Path
import stat
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import URLError
import zipfile

from rsi_researcher.datasets import fetch


class Response(BytesIO):
    def __init__(self, data, status=200, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {} if content_range is None else {"Content-Range": content_range}


def zip_bytes(files):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def tar_bytes(files, symlink=False):
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, BytesIO(data))
        if symlink:
            info = tarfile.TarInfo("link")
            info.type, info.linkname = tarfile.SYMTYPE, "outside"
            archive.addfile(info)
    return buffer.getvalue()


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def use_archive(self, dataset, data):
        spec = fetch.DATASETS[dataset] | {"archive_bytes": len(data)}
        algorithm = "sha256" if dataset == "coat" else "md5"
        spec[algorithm] = (sha256 if algorithm == "sha256" else md5)(data).hexdigest()
        self.patch_spec = patch.dict(fetch.DATASETS, {dataset: spec})
        self.patch_spec.start()
        self.addCleanup(self.patch_spec.stop)
        return spec

    def test_complete_response_extracts_and_verified_manifest_reuses_without_network(self):
        data = zip_bytes({"coat/train.ascii": b"1 0\n", "coat/test.ascii": b"0 3\n"})
        spec = self.use_archive("coat", data)
        with patch.object(fetch, "urlopen", return_value=Response(data)) as network:
            manifest = fetch.fetch_dataset("coat", self.root)
        self.assertEqual(network.call_count, 1)
        self.assertEqual(manifest["sha256"], spec["sha256"])
        self.assertEqual((self.root / "coat/train.ascii").read_bytes(), b"1 0\n")
        with patch.object(fetch, "urlopen", side_effect=AssertionError("network called")):
            self.assertEqual(fetch.fetch_dataset("coat", self.root), manifest)

    def test_existing_legacy_manifest_string_hashes_are_compatible(self):
        data = zip_bytes({"coat/train.ascii": b"4"})
        spec = self.use_archive("coat", data)
        (self.root / "coat.zip").write_bytes(data)
        (self.root / "coat").mkdir()
        (self.root / "coat/train.ascii").write_bytes(b"4")
        previous = {key: spec[key] for key in ("dataset", "download_url", "sha256")}
        previous["files"] = {"coat/train.ascii": sha256(b"4").hexdigest()}
        (self.root / "source.json").write_text(json.dumps(previous), encoding="utf-8")
        with patch.object(fetch, "urlopen", side_effect=AssertionError("network called")):
            self.assertEqual(fetch.fetch_dataset("coat", self.root), previous)

    def test_ranges_resume_only_completed_chunks_after_bounded_retries(self):
        data = tar_bytes({"KuaiRand-Pure/data/log.csv": b"user,item\nu,i\n"})
        self.use_archive("kuairand-pure", data)
        requests = []

        def network(request, **kwargs):
            start, end = map(int, request.get_header("Range")[6:].split("-"))
            requests.append(start)
            if start:
                raise URLError("temporary timeout")
            return Response(data[start:end + 1], 206, f"bytes {start}-{end}/{len(data)}")

        with patch.object(fetch, "CHUNK_BYTES", 32), patch.object(fetch, "urlopen", side_effect=network):
            with self.assertRaisesRegex(RuntimeError, "rerun to resume"):
                fetch.fetch_dataset("kuairand-pure", self.root)
        self.assertEqual(requests, [0, 32, 32, 32])
        self.assertEqual((self.root / "KuaiRand-Pure.tar.gz.partial").stat().st_size, 32)
        resumed = []

        def resumed_network(request, **kwargs):
            start, end = map(int, request.get_header("Range")[6:].split("-"))
            resumed.append(start)
            return Response(data[start:end + 1], 206, f"bytes {start}-{end}/{len(data)}")

        with patch.object(fetch, "CHUNK_BYTES", 32), patch.object(fetch, "urlopen", side_effect=resumed_network):
            manifest = fetch.fetch_dataset("kuairand-pure", self.root)
        self.assertEqual(resumed[0], 32)
        self.assertEqual(manifest["md5"], md5(data).hexdigest())
        self.assertEqual((self.root / "KuaiRand-Pure/data/log.csv").read_bytes(), b"user,item\nu,i\n")
        self.assertFalse((self.root / "KuaiRand-Pure.tar.gz.partial.json").exists())

    def test_range_header_mismatch_is_rejected_before_append(self):
        data = zip_bytes({"coat/a": b"1"})
        self.use_archive("coat", data)
        with patch.object(fetch, "urlopen", return_value=Response(data, 206, "bytes 0-1/2")):
            with self.assertRaisesRegex(ValueError, "Content-Range"):
                fetch.fetch_dataset("coat", self.root)
        self.assertEqual((self.root / "coat.zip.partial").stat().st_size, 0)

    def test_bad_archive_checksum_never_extracts(self):
        data = zip_bytes({"coat/a": b"1"})
        spec = self.use_archive("coat", data)
        spec["sha256"] = "0" * 64
        with patch.object(fetch, "urlopen", return_value=Response(data)):
            with self.assertRaisesRegex(ValueError, "checksum"):
                fetch.fetch_dataset("coat", self.root)
        self.assertFalse((self.root / "coat").exists())

    def test_unknown_existing_archive_and_partial_are_not_overwritten(self):
        (self.root / "coat.zip").write_bytes(b"user file")
        with patch.object(fetch, "urlopen", side_effect=AssertionError("network called")):
            with self.assertRaises(ValueError):
                fetch.fetch_dataset("coat", self.root)
        self.assertEqual((self.root / "coat.zip").read_bytes(), b"user file")
        other = self.root / "another"
        other.mkdir()
        (other / "coat.zip.partial").write_bytes(b"unknown")
        with self.assertRaises(FileExistsError):
            fetch.fetch_dataset("coat", other)
        self.assertEqual((other / "coat.zip.partial").read_bytes(), b"unknown")

    def test_existing_extracted_files_are_not_overwritten(self):
        data = zip_bytes({"coat/a": b"official"})
        self.use_archive("coat", data)
        (self.root / "coat.zip").write_bytes(data)
        (self.root / "coat").mkdir()
        (self.root / "coat/a").write_bytes(b"user change")
        with self.assertRaisesRegex(FileExistsError, "different file"):
            fetch.fetch_dataset("coat", self.root)
        self.assertEqual((self.root / "coat/a").read_bytes(), b"user change")

    def test_archive_traversal_and_links_are_rejected_before_any_extraction(self):
        unsafe = zip_bytes({"coat/ok": b"yes", "../outside": b"no"})
        archive = self.root / "unsafe.zip"
        archive.write_bytes(unsafe)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            fetch._extract(archive, self.root)
        self.assertFalse((self.root / "coat").exists())
        archive = self.root / "unsafe.tar.gz"
        archive.write_bytes(tar_bytes({"KuaiRand-Pure/ok": b"yes"}, symlink=True))
        with self.assertRaisesRegex(ValueError, "link or special"):
            fetch._extract(archive, self.root)
        self.assertFalse((self.root / "KuaiRand-Pure").exists())
        archive = self.root / "symlink.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            handle.writestr(info, "outside")
        with self.assertRaisesRegex(ValueError, "link or special"):
            fetch._extract(archive, self.root)

    def test_modified_manifest_files_and_unknown_source_are_rejected(self):
        data = zip_bytes({"coat/a": b"official"})
        self.use_archive("coat", data)
        with patch.object(fetch, "urlopen", return_value=Response(data)):
            fetch.fetch_dataset("coat", self.root)
        (self.root / "coat/a").write_bytes(b"modified")
        with patch.object(fetch, "urlopen", side_effect=AssertionError("network called")):
            with self.assertRaisesRegex(ValueError, "recorded checksum"):
                fetch.fetch_dataset("coat", self.root)
        source = self.root / "source.json"
        source.write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            fetch.fetch_dataset("coat", self.root)
        self.assertEqual(source.read_text(encoding="utf-8"), "{}")


if __name__ == "__main__":
    unittest.main()
