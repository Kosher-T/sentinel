import unittest
import sys
import numpy as np
from pathlib import Path

# Project setup
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from detector_data_drift.analyzer import (
    analyze_drift,
    _compute_drift_score,
    _bootstrap_confidence_interval,
)


class TestConfidenceIntervals(unittest.TestCase):
    """Tests for bootstrap-based confidence intervals in drift analysis."""

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 20
        # Identical distributions (no drift)
        self.baseline = np.random.randn(self.n_samples, self.n_features)
        self.identical = self.baseline.copy()
        # Shifted distribution (clear drift)
        self.shifted = self.baseline + 3.0

    # ----- analyze_drift returns 4-tuple -----

    def test_analyze_drift_returns_four_values(self):
        """analyze_drift now returns (prob, metrics, root_cause, ci)."""
        result = analyze_drift(self.baseline, self.identical, n_bootstrap=20)
        self.assertEqual(len(result), 4)

    def test_ci_dict_has_expected_keys(self):
        """CI dict contains 'low', 'high', and 'margin'."""
        _, _, _, ci = analyze_drift(self.baseline, self.identical, n_bootstrap=20)
        self.assertIn("low", ci)
        self.assertIn("high", ci)
        self.assertIn("margin", ci)

    # ----- CI bounds order -----

    def test_ci_bounds_order(self):
        """low <= high for any input."""
        _, _, _, ci = analyze_drift(self.baseline, self.shifted, n_bootstrap=30)
        self.assertLessEqual(ci["low"], ci["high"])

    def test_ci_margin_nonnegative(self):
        """margin is always >= 0."""
        _, _, _, ci = analyze_drift(self.baseline, self.shifted, n_bootstrap=30)
        self.assertGreaterEqual(ci["margin"], 0)

    # ----- Identical distributions → tight CI -----

    def test_identical_distributions_narrow_ci(self):
        """When distributions are identical, CI margin should be very small."""
        _, _, _, ci = analyze_drift(self.baseline, self.identical, n_bootstrap=50)
        # Margin as a probability (0-1), should be tiny for identical data
        self.assertLess(ci["margin"], 0.05, "CI margin should be tight for identical distributions")

    # ----- Shifted distributions → positive drift -----

    def test_shifted_distributions_positive_drift(self):
        """When distributions are shifted, CI low should be above 0."""
        prob, _, _, ci = analyze_drift(self.baseline, self.shifted, n_bootstrap=50)
        self.assertGreater(ci["low"], 0, "CI low should be positive for drifted data")
        self.assertGreater(prob, 0, "Drift probability should be positive")

    # ----- _compute_drift_score -----

    def test_compute_drift_score_returns_float(self):
        """_compute_drift_score returns a single float."""
        score = _compute_drift_score(self.baseline, self.identical)
        self.assertIsInstance(score, float)

    def test_compute_drift_score_range(self):
        """Score should be between 0 and 1."""
        score = _compute_drift_score(self.baseline, self.shifted)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_compute_drift_score_identical_is_low(self):
        """Score for identical distributions should be near 0."""
        score = _compute_drift_score(self.baseline, self.identical)
        self.assertLess(score, 0.05)

    def test_compute_drift_score_shifted_is_high(self):
        """Score for shifted distributions should be measurably high."""
        score = _compute_drift_score(self.baseline, self.shifted)
        self.assertGreater(score, 0.1)

    # ----- _bootstrap_confidence_interval -----

    def test_bootstrap_returns_dict(self):
        """Bootstrap function returns a dict with expected keys."""
        ci = _bootstrap_confidence_interval(self.baseline, self.identical, n_bootstrap=20)
        self.assertIsInstance(ci, dict)
        self.assertIn("low", ci)
        self.assertIn("high", ci)
        self.assertIn("margin", ci)

    def test_bootstrap_small_sample_still_works(self):
        """Bootstrap works even with small sample sizes."""
        small_base = np.random.randn(5, 10)
        small_curr = np.random.randn(5, 10)
        ci = _bootstrap_confidence_interval(small_base, small_curr, n_bootstrap=10)
        self.assertIn("low", ci)
        self.assertLessEqual(ci["low"], ci["high"])


if __name__ == '__main__':
    unittest.main()
