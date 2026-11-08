import json
from html import unescape
from pathlib import Path
import re
import unittest
from urllib.parse import parse_qs, unquote, urlsplit

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
        page = (ROOT / "index.html").read_text()
        match = re.search(r'<script id="document-data" type="application/json">(.*?)</script>',
                          page, re.S)
        self.assertIsNotNone(match)
        documents = json.loads(match[1])
        self.assertEqual(documents, render_documents(ROOT), "Run python scripts/build_demo.py")
        pages = {"index.html": page, **{name: doc["html"] for name, doc in documents.items()}}
        for name, markup in pages.items():
            for href in re.findall(r'href="([^"]+)"', markup):
                url = urlsplit(unescape(href))
                if url.scheme or url.netloc:
                    continue
                with self.subTest(document=name, link=href):
                    if url.path:
                        self.assertTrue((ROOT / unquote(url.path)).exists())
                    for target in parse_qs(url.fragment).get("doc", []):
                        self.assertIn(target, documents)
