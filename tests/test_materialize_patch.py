import tempfile
import unittest
from pathlib import Path

from materialize_patch import source_checkpoint
from pilot import sha256


class PortableCheckpointTests(unittest.TestCase):
    def test_relocated_checkpoint_must_match_the_pinned_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "base.safetensors"
            source.write_bytes(b"original checkpoint")
            manifest = {"source_checkpoint": "/old/location", "source_checkpoint_sha256": sha256(source)}
            self.assertEqual(source_checkpoint(manifest, source), source)
            source.write_bytes(b"wrong checkpoint")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                source_checkpoint(manifest, source)
            with self.assertRaisesRegex(ValueError, "requires its pinned"):
                source_checkpoint({"source_checkpoint": "/old/location"}, source)
            self.assertEqual(source_checkpoint({"source_checkpoint": str(source)}), source)


if __name__ == "__main__":
    unittest.main()
