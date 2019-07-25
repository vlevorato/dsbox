import unittest

import numpy as np
import pandas as pd
from dsbox.ml.calibration import BinaryCalibrator
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class BinaryCalibratorTest(unittest.TestCase):
    def test_params(self):
        X = pd.DataFrame(np.arange(0, 100))
        y = np.zeros(100)
        y[2] = 1
        y[3] = 1

        bin_calib = BinaryCalibrator(RandomForestClassifier(n_estimators=10))
        bin_calib.fit(X, y)

        self.assertEqual(bin_calib.major_class_amount_, 98)
        self.assertEqual(bin_calib.minor_class_amount_, 2)
        self.assertEqual(bin_calib.beta_, 0.020408163265306124)
        self.assertEqual(bin_calib.tau_s_, 0.5)
        self.assertEqual(bin_calib.bias_corrected_threshold(), 0.02)

    def test_undersampling(self):
        X = pd.DataFrame(np.arange(0, 10))
        y = np.zeros(10)
        y[2] = 1
        y[3] = 1

        bin_calib = BinaryCalibrator(RandomForestClassifier(n_estimators=10))
        bin_calib.fit(X, y)

        self.assertEqual(len(bin_calib.X_), 4)

    def test_estimator(self):
        # set random seed for test
        seed = 42
        np.random.seed(seed)

        X, y = make_classification(n_samples=30, n_features=2, n_redundant=0, n_informative=2,
                                   random_state=seed, weights=[0.95, 0.05])

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.2, random_state=seed)

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

        bin_calib = BinaryCalibrator(RandomForestClassifier(random_state=seed, n_estimators=10))
        bin_calib.fit(X_train, y_train)

        y_pred = bin_calib.predict_proba(X_test)

        y_to_pred = np.asmatrix([[0.90789474, 0.09210526],
                                 [0.97183099, 0.02816901],
                                 [0.99519231, 0.00480769],
                                 [0.97183099, 0.02816901],
                                 [0.97183099, 0.02816901],
                                 [0.97183099, 0.02816901]])

        y_pred = np.round(y_pred, 6)
        y_to_pred = np.round(y_to_pred, 6)

        diff_pred = y_pred - y_to_pred

        self.assertEqual(np.sum(diff_pred), 0.0)

        y_pred = bin_calib.predict(X_test)
        self.assertEqual(np.sum(y_pred), 0.0)
