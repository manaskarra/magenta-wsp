import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from compare_piano import load_cases, embed_case


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


if __name__ == "__main__":
    unittest.main()
