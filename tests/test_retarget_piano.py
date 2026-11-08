import unittest

import numpy as np

from retarget_piano import retarget_delta


class RetargetTests(unittest.TestCase):
    def test_maps_target_preserves_controls_and_unused_subspace(self):
        rng = np.random.default_rng(18)
        basis, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        source, preserve = basis[:3], basis[3:7]
        destination = rng.normal(size=source.shape)
        weight = rng.normal(size=(12, 5))
        delta = retarget_delta(weight, source, destination, preserve)
        np.testing.assert_allclose(source @ (weight + delta), destination @ weight, atol=1e-10)
        np.testing.assert_allclose(preserve @ delta, 0, atol=1e-10)
        np.testing.assert_allclose(basis[7:] @ delta, 0, atol=1e-10)

    def test_rejects_invalid_calibration(self):
        weight = np.ones((4, 2))
        for source, destination, preserve in [
            (np.ones((0, 4)), np.ones((0, 4)), np.ones((1, 4))),
            (np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 4))),
            (np.ones((1, 4)), np.full((1, 4), np.nan), np.ones((1, 4))),
        ]:
            with self.subTest(source_shape=source.shape):
                with self.assertRaises(ValueError):
                    retarget_delta(weight, source, destination, preserve)


if __name__ == "__main__":
    unittest.main()
