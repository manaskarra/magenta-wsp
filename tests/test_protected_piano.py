import unittest
import numpy as np
from protected_piano import protected_gate


class ProtectedGateTests(unittest.TestCase):
    def test_preserves_audio_and_instrument_means_with_consistent_piano_response(self):
        rng = np.random.default_rng(91)
        piano = rng.normal(size=(20, 32)); piano[:, 0] += 3
        controls = rng.normal(size=(3, 20, 32)); controls[:, :, 0] -= 1
        audio_piano = rng.normal(size=(4, 32)); audio_piano[:, 0] += 3
        audio_controls = rng.normal(size=(4, 32)); audio_controls[:, 0] -= 1
        gate = protected_gate(piano, controls, audio_piano, audio_controls)
        np.testing.assert_allclose([piano.mean(0) @ gate, audio_piano.mean(0) @ gate], 1, atol=1e-8)
        np.testing.assert_allclose(controls.mean(1) @ gate, 0, atol=1e-8)
        np.testing.assert_allclose(audio_controls @ gate, 0, atol=1e-8)
        # An exact preservation input gets no rank-one weight-update response.
        update = np.outer(gate, rng.normal(size=8))
        np.testing.assert_allclose(audio_controls @ update, 0, atol=1e-8)

    def test_rejects_invalid_or_inconsistent_constraints(self):
        p = np.ones((3, 8)); c = np.ones((2, 3, 8)); a = np.ones((2, 8))
        for ridge in (0, -1, float("nan"), 1):
            with self.subTest(ridge=ridge), self.assertRaises(ValueError):
                protected_gate(p, c, a, a, ridge)


if __name__ == "__main__":
    unittest.main()
