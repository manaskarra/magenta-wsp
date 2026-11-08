"""Refresh the static document reader and checksums from a Git checkout."""

import json
from pathlib import Path
import re
import subprocess

from common import sha256
from demo_documents import render_documents

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = ROOT / "index.html"
    documents = json.dumps(render_documents(ROOT), ensure_ascii=False)
    documents = documents.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    html, count = re.subn(
        r'(<script id="document-data" type="application/json">).*?(</script>)',
        lambda match: match[1] + documents + match[2], index.read_text(), flags=re.S,
    )
    if count != 1:
        raise ValueError("Expected one document-data block in index.html")
    index.write_text(html)
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
    ).decode().split("\0")
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = {
        name: {"bytes": (ROOT / name).stat().st_size, "sha256": sha256(ROOT / name)}
        for name in sorted(set(names))
        if name and name != "manifest.json" and (ROOT / name).is_file()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Updated document reader and checksums for {len(manifest['files'])} files.")


if __name__ == "__main__":
    main()
