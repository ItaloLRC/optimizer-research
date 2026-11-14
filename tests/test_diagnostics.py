from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from optimizer_research.diagnostics import (
    counter_diagnostics, pooled_window, threshold_bounds, first_sustained, curve_summary,
    boost_diagnostics,
)


class DiagnosticTests(unittest.TestCase):
    def test_incremental_exposure_weighting_and_disjoint_bins(self):
        d = counter_diagnostics([1, 9], [2, 12], [2, 20])
        np.testing.assert_array_equal(d['increments'], [[1, 8], [2, 10], [2, 18]])
        np.testing.assert_allclose(d['pooled_rates'], [.45, .15, .4])
        np.testing.assert_allclose(pooled_window(d, 0, 2), [.45, .15, .4])
        self.assertNotEqual(np.mean(d['rates'][0]), .45)
        np.testing.assert_allclose(d['rates'].sum(axis=0), 1)

    def test_cumulative_nesting_does_not_guarantee_incremental_nesting(self):
        with self.assertRaises(ValueError):
            counter_diagnostics([1, 5], [4, 6], [5, 10])

    def test_zero_exposure_remains_missing(self):
        d = counter_diagnostics([0, 1], [0, 2], [0, 4])
        self.assertTrue(np.isnan(d['rates'][:, 0]).all())
        self.assertTrue(np.isnan(pooled_window(d, 0, 1)).all())

    def test_rejects_noninteger_decreasing_and_mismatched_counts(self):
        for args in (([1.], [2], [3]), ([2, 1], [3, 4], [5, 6]),
                     ([1], [2, 3], [4]), ([-1], [2], [3])):
            with self.assertRaises(ValueError):
                counter_diagnostics(*args)

    def test_bounds_enclose_every_compatible_empirical_distribution(self):
        raw = np.array([.1, .2, .3, .45, .6, .8])
        bounds = threshold_bounds(2/6, 4/6, [1, 2, 3, 4, 8])
        for b in bounds:
            actual = np.mean(raw <= b['threshold'])
            self.assertLessEqual(b['lower'], actual)
            self.assertLessEqual(actual, b['upper'])
        self.assertEqual([b['identified'] for b in bounds], [False, True, False, True, False])

    def test_target_uses_entire_sustained_window_and_censoring(self):
        self.assertEqual(first_sustained([.2, .1, .09, .08], .1, 3), 2)
        self.assertIsNone(first_sustained([.2, .09, .2, .09], .1, 2))
        self.assertIsNone(first_sustained([.09], .1, 2))

    def test_auc_and_detrending_on_analytical_linear_curve(self):
        s = curve_summary([4, 3, 2, 1], 4)
        self.assertEqual(s['auc'], 7.5)
        self.assertEqual(s['normalized_auc'], 2.5)
        self.assertAlmostEqual(s['late_slope'], -1)
        self.assertAlmostEqual(s['detrended_rms'], 0)
        self.assertGreater(s['temporal_sd'], 0)

    def test_historical_counters_are_nested_and_complete(self):
        root = Path(__file__).resolve().parents[1]/'data/original'
        arrays = [np.load(root/f'AdaSym_fminist_{s}.npy', allow_pickle=False) for s in ('c4','c2','full')]
        d = boost_diagnostics(*arrays)
        self.assertEqual(d['increments'].shape, (3, 150))
        np.testing.assert_allclose(d['within_c2'].sum(axis=0), 1)
        self.assertEqual(sum(d['increments'][2]), int(arrays[2][-1]))

    def test_full_boost_counter_is_not_a_population_denominator(self):
        d = boost_diagnostics([1, 3], [2, 5], [0, 1])
        np.testing.assert_array_equal(d['increments'], [[1,2],[2,3],[0,1]])
        np.testing.assert_allclose(d['within_c2'][0], [.5, 2/3])


if __name__ == '__main__':
    unittest.main()
