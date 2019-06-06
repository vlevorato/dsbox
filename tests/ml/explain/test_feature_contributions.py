import unittest

import numpy as np
import pandas as pd
from dsbox.ml.explain.feature_contributions import FeatureContributions
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class FeatureContributionsTest(unittest.TestCase):
    def test_feature_contributions(self):
        # set random seed for test
        seed = 42
        np.random.seed(seed)

        X, y = make_classification(n_samples=30, n_features=3, n_redundant=0, n_informative=2, random_state=seed)

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.2, random_state=seed)

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

        clf = RandomForestClassifier(n_estimators=10, random_state=seed)

        clf.fit(X_train, y_train)

        fc = FeatureContributions(clf)
        predictions, bias, contributions = fc.predict(X_test)

        expected_contributions = np.array([
            [[-0.00791667, 0.00791667],
             [-0.23905303, 0.23905303],
             [0.28863636, -0.28863636]],

            [[0.06117424, -0.06117424],
             [-0.06981061, 0.06981061],
             [-0.34969697, 0.34969697]],

            [[0.11117424, -0.11117424],
             [0.08018939, -0.08018939],
             [0.35030303, -0.35030303]],

            [[0.11117424, -0.11117424],
             [-0.01981061, 0.01981061],
             [-0.34969697, 0.34969697]],

            [[0.06117424, -0.06117424],
             [-0.06981061, 0.06981061],
             [-0.34969697, 0.34969697]],

            [[0.06117424, -0.06117424],
             [0.03018939, -0.03018939],
             [0.35030303, -0.35030303]]])

        contributions = np.round(contributions, decimals=7)
        expected_contributions = np.round(expected_contributions, decimals=7)

        self.assertEqual(np.array_equal(contributions, expected_contributions), True)
