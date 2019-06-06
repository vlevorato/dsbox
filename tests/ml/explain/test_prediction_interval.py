import unittest

import numpy as np
from dsbox.ml.explain.prediction_interval import compute_prediction_interval
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# set random seed for test
np.random.seed(42)


class PredictionIntervalTest(unittest.TestCase):
    def test_90_percentile_prediction_interval_has_90_percent_of_ground_truth_inside_the_interval(self):
        # Given
        boston = load_boston()
        data_train, data_eval, target_train, target_eval = train_test_split(
            boston['data'][:100], boston['target'][:100], test_size=0.3, random_state=42)
        rf = RandomForestRegressor(n_estimators=80, n_jobs=-1, random_state=42)
        rf.fit(data_train, target_train)

        # When
        lower_error_list, upper_error_list = compute_prediction_interval(rf, data_eval, percentile=90)

        # Then
        ground_truth_inside_interval = 0
        for i, target_value in enumerate(target_eval):
            if lower_error_list[i] <= target_value <= upper_error_list[i]:
                ground_truth_inside_interval += 1
        ground_truth_inside_interval_percentage = ground_truth_inside_interval / float(len(target_eval))
        # check that we have roughly 90% of ground truth value inside the computed interval
        self.assertAlmostEqual(ground_truth_inside_interval_percentage, 0.90, 2)

    def test_compute_prediction_interval_with_stub_model(self):
        # Given
        class TreeStub:
            def predict(self, data_row):
                return [1]

        class RfStub:
            estimators_ = [TreeStub() for _ in range(100)]

        data_eval = np.array([[1, 2, 3], [1, 2, 4]])
        expected_lower_error_list = [1, 1]
        expected_upper_error_list = [1, 1]

        # When
        lower_error_list, upper_error_list = compute_prediction_interval(RfStub(), data_eval, percentile=90)

        # Then
        self.assertEqual(lower_error_list, expected_lower_error_list)
        self.assertEqual(upper_error_list, expected_upper_error_list)
