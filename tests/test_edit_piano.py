import unittest

import numpy as np

from edit_piano import remove_direction


class ProjectionTests(unittest.TestCase):
    def test_removes_target_component_and_preserves_orthogonal_inputs(self):
        rng = np.random.default_rng(7)
        weight = rng.normal(size=(12, 5)).astype(np.float32)
        direction = rng.normal(size=12)
        unit = direction / np.linalg.norm(direction)
        orthogonal = rng.normal(size=12)
        orthogonal -= (orthogonal @ unit) * unit
        edited = remove_direction(weight, direction)
        self.assertEqual(edited.dtype, weight.dtype)
        np.testing.assert_allclose(unit @ edited, 0, atol=1e-6)
        np.testing.assert_allclose(orthogonal @ edited, orthogonal @ weight, atol=1e-6)
        with self.assertRaises(ValueError):
            remove_direction(weight, np.zeros(12))


if __name__ == "__main__":
    unittest.main()
