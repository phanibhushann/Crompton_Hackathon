import unittest

import numpy as np

from src.models.blend import blend_predictions


class TestBlendPredictions(unittest.TestCase):
    def test_blend_normalizes_weights(self) -> None:
        components = {
            "hierarchical": np.array([10.0, 10.0]),
            "lgbm": np.array([0.0, 0.0]),
        }
        weights = {"hierarchical": 2.0, "lgbm": 1.0}
        # Normalized weights: 2/3 and 1/3 => result = (2/3)*10 + (1/3)*0 = 6.666...
        out = blend_predictions(components, weights)
        self.assertTrue(np.allclose(out, np.array([2.0 * 10.0 / 3.0] * 2)))


if __name__ == "__main__":
    unittest.main()

