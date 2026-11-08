import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from common import sha256
from compare_piano import load_cases, embed_case, main


class EvaluationCaseTests(unittest.TestCase):
    def test_manifest_preserves_pairing_and_rejects_unsafe_or_invalid_cases(self):
        case = {"prompt": "Solo piano", "seed": 11, "style_mapper_seed": 37, "target": "piano"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            valid = {"piano_11": case}
            path.write_text(json.dumps(valid))
            self.assertEqual(load_cases(path), valid)
            for invalid in ({"../escape": case}, {}, {"piano": {**case, "seed": True}},
                            {"piano": {**case, "style_mapper_seed": -1}},
                            {"piano": {**case, "prompt": " "}},
                            {"piano": {**case, "unknown": 1}}):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    path.write_text(json.dumps(invalid))
                    load_cases(path)

    def test_audio_cases_pin_the_reference_without_text_conditioning(self):
        case = {"audio_reference": "/tmp/reference.wav", "audio_sha256": "a" * 64,
                "seed": 53, "target": "piano"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps({"audio": case}))
            self.assertEqual(load_cases(path), {"audio": case})
            for changed in ({"audio_reference": "relative.wav"}, {"audio_sha256": "bad"},
                            {"prompt": "piano"}, {"seed": False}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    path.write_text(json.dumps({"audio": {**case, **changed}}))
                    load_cases(path)

    def test_conditioning_excludes_the_evaluation_label_and_checks_audio_integrity(self):
        model = Mock()
        case = {"prompt": "Solo piano", "style_mapper_seed": 23, "seed": 5, "target": "flute"}
        embed_case(model, case)
        model.embed_style.assert_called_once_with("Solo piano", use_mapper=True, seed=23)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.wav"; path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                embed_case(model, {"audio_reference": str(path), "audio_sha256": "0" * 64})

    def test_paired_generation_uses_declared_files_without_a_research_archive(self):
        """Exercise the command flow with a stub runtime, not musical quality."""
        modules = {name: ModuleType(name) for name in
                   ("mlx", "mlx.core", "huggingface_hub", "magenta_rt", "magenta_rt.config")}
        mx = modules["mlx.core"]
        mx.set_cache_limit, mx.clear_cache, mx.random = Mock(), Mock(), SimpleNamespace(seed=Mock())
        modules["mlx"].core = mx
        modules["magenta_rt.config"].MUSICCOCA = SimpleNamespace(key="style")
        model = Mock(temperature=1, top_k=100, cfg_scales=[1])
        waveform = SimpleNamespace(samples=np.full((20, 2), 0.1, dtype=np.float32), sample_rate=10,
                                   write=lambda path: Path(path).write_bytes(b"fixture audio"))
        model.generate.return_value = (waveform, None)
        runtime = modules["magenta_rt"]
        runtime.MagentaRT2Mlx, runtime.paths = Mock(return_value=model), Mock()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = root / "cases.json"
            cases.write_text(json.dumps({"piano": {
                "prompt": "Solo piano", "seed": 11, "style_mapper_seed": 37, "target": "piano",
            }}))
            checkpoint = root / "edited.safetensors"
            checkpoint.write_bytes(b"fixture checkpoint")
            edit_record = root / "edit.json"
            edit_record.write_text(json.dumps({"checkpoint": str(checkpoint),
                                               "checkpoint_sha256": sha256(checkpoint)}))
            modules["huggingface_hub"].snapshot_download = Mock(return_value=str(root / "snapshot"))
            args = ["compare_piano.py", "--cases", str(cases), "--output", str(root / "output")]
            with patch.dict(sys.modules, modules), patch("builtins.print"):
                for extra in ([], ["--edited", "--edit-record", str(edit_record)]):
                    with patch.object(sys, "argv", args + extra):
                        main()
                with patch.object(sys, "argv", args), self.assertRaises(FileExistsError):
                    main()
            for variant in ("baseline", "edited"):
                record = json.loads((root / "output" / variant / "piano.json").read_text())
                self.assertEqual(record["seed"], 11)
                self.assertEqual(record["style_mapper_seed"], 37)
                self.assertEqual(record["cases_sha256"], sha256(cases))
                self.assertEqual(record["prompt"], "Solo piano")
            self.assertEqual([call.kwargs["checkpoint"] for call in runtime.MagentaRT2Mlx.call_args_list],
                             [str(root / "snapshot/checkpoints/mrt2_base.safetensors"), str(checkpoint)])
            self.assertEqual([call.args for call in mx.random.seed.call_args_list], [(11,), (11,)])
            self.assertEqual(model.generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
