import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.study import (
    CALIBRATION_DAYS, FEE_BPS, SLIPPAGE_BPS,
    basket_index, simulate, walk_forward_probabilities,
)


class StudyAccountingTests(unittest.TestCase):
    def test_basket_resets_to_equal_weight_each_day(self):
        prices = pd.DataFrame({"A": [100, 200, 100], "B": [100, 100, 200]})
        index = basket_index(prices, ("A", "B"))
        self.assertAlmostEqual(index.iloc[0], 1)
        self.assertAlmostEqual(index.iloc[1], 1.5)
        self.assertAlmostEqual(index.iloc[2], 1.875)

    def test_close_signal_waits_until_after_next_close(self):
        dates = pd.bdate_range("2020-01-01", periods=510)
        tech = pd.Series(100 * 1.01 ** np.arange(510), index=dates)
        infra = pd.Series(100.0, index=dates)
        signal = pd.Series(0, index=dates)
        signal.iloc[CALIBRATION_DAYS:] = 1
        result = simulate(signal, tech, infra, beta=1, first_test_index=CALIBRATION_DAYS)
        first = result.iloc[0]
        self.assertEqual(first["position"], 1)
        self.assertAlmostEqual(first["gross_return"], 0.005)
        self.assertAlmostEqual(first["trading_cost"], (FEE_BPS + SLIPPAGE_BPS) / 10_000)
        self.assertGreater(first["borrow_cost"], 0)

    def test_walk_forward_never_fits_on_a_label_reaching_test_period(self):
        dates = pd.bdate_range("2020-01-01", periods=700)
        features = pd.DataFrame({"z": np.arange(700)}, index=dates)
        labels = pd.Series(np.arange(700) % 2, index=dates)
        last_train_dates = []

        class SpyForest:
            def __init__(self, **kwargs):
                pass

            def fit(self, x, y):
                last_train_dates.append(x.index[-1])

            def predict_proba(self, x):
                return np.tile([0.5, 0.5], (len(x), 1))

        with patch("src.study.RandomForestClassifier", SpyForest):
            predictions = walk_forward_probabilities(features, labels)
        self.assertTrue(last_train_dates)
        self.assertLessEqual(last_train_dates[0], dates[CALIBRATION_DAYS - 7])
        self.assertTrue(predictions.iloc[:CALIBRATION_DAYS].isna().all())
        self.assertTrue(predictions.iloc[CALIBRATION_DAYS:].notna().any())


if __name__ == "__main__":
    unittest.main()
