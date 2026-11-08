import json
from pathlib import Path
import re
import unittest

from common import sha256
from demo_documents import render_documents

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_manifest_matches_published_files_and_patch_records(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            with self.subTest(file=name):
                path = ROOT / name
                self.assertEqual(path.stat().st_size, expected["bytes"])
                self.assertEqual(sha256(path), expected["sha256"])
        for directory, candidate in (("patch", "stronger"), ("experimental", "gatefix")):
            record = json.loads((ROOT / directory / "edit.json").read_text())
            self.assertEqual(sha256(ROOT / directory / f"{candidate}.safetensors"),
                             record["candidates"][candidate]["patch_sha256"])

    def test_embedded_documents_match_markdown_and_local_links_exist(self):
        match = re.search(r'<script id="document-data" type="application/json">(.*?)</script>',
                          (ROOT / "index.html").read_text(), re.S)
        self.assertIsNotNone(match)
        documents = json.loads(match[1])
        self.assertEqual(documents, render_documents(ROOT), "Run python scripts/build_demo.py")
        from urllib.parse import unquote, urlsplit
        for name, document in documents.items():
            for href in re.findall(r'href="([^"]+)"', document["html"]):
                url = urlsplit(href)
                if not url.scheme and not url.netloc and url.path:
                    with self.subTest(document=name, link=href):
                        self.assertTrue((ROOT / unquote(url.path)).exists())
