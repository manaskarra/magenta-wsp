import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pilot import audio_metrics, completed_case, sha256


class PilotTests(unittest.TestCase):
    def test_audio_checks_measure_actual_duration(self):
        audio = np.full((44100, 2), 0.1, dtype=np.float32)
        metrics = audio_metrics(audio, 44100)
        self.assertEqual(metrics["duration_seconds"], 1)
        self.assertAlmostEqual(metrics["rms_dbfs"], -20, places=5)

    def test_invalid_or_silent_audio_cannot_count_as_success(self):
        for audio in (np.empty((0, 2)), np.ones(20), np.zeros((20, 2)), np.full((20, 2), np.nan)):
            with self.subTest(shape=audio.shape), self.assertRaises(ValueError):
                audio_metrics(audio, 44100)

    def test_resume_checks_request_and_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "case"
            request = {"seed": 7, "revision": "pinned"}
            self.assertFalse(completed_case(run, request))
            run.mkdir()
            with self.assertRaises(RuntimeError):
                completed_case(run, request)
            wav = run / "audio.wav"
            wav.write_bytes(b"test audio artifact")
            (run / "result.json").write_text(json.dumps({"request": request, "wav_sha256": sha256(wav)}))
            self.assertTrue(completed_case(run, request))
            with self.assertRaises(RuntimeError):
                completed_case(run, {**request, "seed": 8})
            wav.write_bytes(b"modified")
            with self.assertRaises(RuntimeError):
                completed_case(run, request)


if __name__ == "__main__":
    unittest.main()
