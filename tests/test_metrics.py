import unittest

import numpy as np

from src.metrics import compute_rmse, compute_wape


class TestMetrics(unittest.TestCase):
    def test_compute_wape_basic(self) -> None:
        actual = np.array([1.0, 1.0])
        predicted = np.array([0.0, 2.0])
        # abs errors sum = 1 + 1 = 2 ; actual sum = 2 => WAPE = 1.0
        self.assertAlmostEqual(compute_wape(actual, predicted), 1.0)

    def test_compute_rmse_basic(self) -> None:
        actual = np.array([1.0, -1.0])
        predicted = np.array([0.0, 0.0])
        # squared errors = 1 and 1 => mean 1 => RMSE 1
        self.assertAlmostEqual(compute_rmse(actual, predicted), 1.0)


if __name__ == "__main__":
    unittest.main()

