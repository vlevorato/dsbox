import unittest
import numpy as np

from dsbox.ml.ensemble.rrclassifier.randomrotation import random_rotation_matrix
from dsbox.ml.ensemble.rrclassifier import RRForestClassifier, RRExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class TestRRClassifier(unittest.TestCase):

    def test_random_rotation(self):
        R = random_rotation_matrix(3)

        self.assertTrue(np.allclose(R.T, np.linalg.inv(R)))

        self.assertIsNotNone(np.linalg.det(R))

    def test_RRForestClassifier(self):
        np.random.seed(42)

        # given
        X = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(X.data, X.target, test_size=0.33, stratify=X.target,
                                                            random_state=42)
        # when
        clf = RRForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # then
        expected_score = 0.9
        f_score = f1_score(y_test, y_pred, average='micro')
        self.assertGreater(f_score, expected_score)

    def test_RRExtraTreesClassifier(self):
        np.random.seed(42)

        # given
        X = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(X.data, X.target, test_size=0.33, stratify=X.target,
                                                            random_state=42)
        # when
        clf = RRExtraTreesClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # then
        expected_score = 0.9
        f_score = f1_score(y_test, y_pred, average='micro')
        self.assertGreater(f_score, expected_score)
