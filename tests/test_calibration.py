import unittest

import numpy as np

from scripts.eval_calibration import compute_ece, compute_mce
from scripts.generate_reliability_plot import compute_reliability_bins


class CalibrationTests(unittest.TestCase):
    def test_perfect_confidence_is_included(self):
        self.assertAlmostEqual(compute_ece([1.0], [0.0]), 1.0)
        self.assertAlmostEqual(compute_mce([1.0], [0.0]), 1.0)
        _, confidence, accuracy, counts = compute_reliability_bins([1.0], [0.0])
        self.assertEqual(int(counts[-1]), 1)
        self.assertAlmostEqual(confidence[-1], 1.0)
        self.assertAlmostEqual(accuracy[-1], 0.0)

    def test_weighted_bin_error(self):
        confidence = [0.2, 0.3, 0.9, 1.0]
        correct = [0, 1, 1, 0]
        self.assertAlmostEqual(compute_ece(confidence, correct, n_bins=2), 0.35)
        self.assertAlmostEqual(compute_mce(confidence, correct, n_bins=2), 0.45)

    def test_invalid_inputs_are_rejected(self):
        for confidence, accuracy in [([], []), ([1.1], [1]), ([np.nan], [1]), ([0.5], [])]:
            with self.subTest(confidence=confidence, accuracy=accuracy):
                with self.assertRaises(ValueError):
                    compute_ece(confidence, accuracy)


if __name__ == "__main__":
    unittest.main()
