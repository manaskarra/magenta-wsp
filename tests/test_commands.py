import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import load_file, save_file

from common import audio_metrics, sha256
from edit import TARGET
from materialize_patch import main as materialize

ROOT = Path(__file__).resolve().parents[1]


class CommandTests(unittest.TestCase):
    def test_help_and_missing_edit_record_work_without_the_model_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("materialize_patch.py", "compare_piano.py", "rebuild_protected_patch.py"):
                with self.subTest(command=name):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "scripts" / name), "--help"],
                        cwd=tmp, capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/compare_piano.py"), "--edited",
                 "--cases", str(ROOT / "demo_cases.json"), "--output", tmp],
                cwd=tmp, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--edited requires --edit-record", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_patch_application_preserves_other_tensors_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, patch_path = root / "base.safetensors", root / "stronger.safetensors"
            original = np.zeros((3, 4), dtype=np.float32)
            edited = np.ones_like(original)
            untouched = np.arange(5, dtype=np.float32)
            save_file({TARGET: original, "untouched": untouched}, base)
            save_file({TARGET: edited}, patch_path)
            manifest = {
                "source_checkpoint": "unavailable/research/base.safetensors",
                "source_checkpoint_sha256": sha256(base), "repo_id": "fixture",
                "revision": "fixture", "calibration": [], "formula": "fixture",
                "candidates": {"stronger": {"patch_sha256": sha256(patch_path)}},
            }
            (root / "edit.json").write_text(json.dumps(manifest))
            output = root / "checkpoint"
            argv = ["materialize_patch.py", "--patch-dir", str(root),
                    "--base-checkpoint", str(base), "--output", str(output)]
            with patch.object(sys, "argv", argv), patch("builtins.print"):
                materialize()
                with self.assertRaises(FileExistsError):
                    materialize()
            result = load_file(output / "mrt2_base.safetensors")
            np.testing.assert_array_equal(result[TARGET], edited)
            np.testing.assert_array_equal(result["untouched"], untouched)
            record = json.loads((output / "edit.json").read_text())
            self.assertEqual(record["changed_tensors"], [TARGET])
            self.assertEqual(record["checkpoint_sha256"], sha256(output / "mrt2_base.safetensors"))
            patch_path.write_bytes(b"damaged patch")
            with patch.object(sys, "argv", [*argv[:-1], str(root / "rejected")]):
                with self.assertRaisesRegex(ValueError, "Patch hash mismatch"):
                    materialize()
            self.assertFalse((root / "rejected").exists())

    def test_audio_checks_measure_duration_and_reject_invalid_output(self):
        metrics = audio_metrics(np.full((44100, 2), 0.1, dtype=np.float32), 44100)
        self.assertEqual(metrics["duration_seconds"], 1)
        self.assertAlmostEqual(metrics["rms_dbfs"], -20, places=5)
        for audio in (np.empty((0, 2)), np.ones(20), np.zeros((20, 2)), np.full((20, 2), np.nan)):
            with self.subTest(shape=audio.shape), self.assertRaises(ValueError):
                audio_metrics(audio, 44100)
