import unittest

from numpy.testing import assert_array_equal

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from sklearn.base import clone

from dsbox.ml.markov.seqlearn import StructuredPerceptron


class TestSeqPerceptron(unittest.TestCase):
    def test_perceptron(self):
        X = [[0, 1, 0],
             [0, 1, 0],
             [1, 0, 0],
             [0, 1, 0],
             [1, 0, 0],
             [0, 0, 1],
             [0, 0, 1],
             [0, 1, 0],
             [1, 0, 0],
             [1, 0, 0]]

        y = [0, 0, 0, 0, 0, 1, 1, 0, 2, 2]

        clf = StructuredPerceptron(verbose=False, random_state=37, max_iter=15)
        clf.fit(X, y, [len(y)])
        assert_array_equal(y, clf.predict(X))

        # Try again with string labels and sparse input.
        y_str = np.array(["eggs", "ham", "spam"])[y]

        clf = clone(clf)
        clf.fit(csc_matrix(X), y_str, [len(y_str)])
        assert_array_equal(y_str, clf.predict(coo_matrix(X)))

        X2 = np.vstack([X, X])
        y2 = np.hstack([y_str, y_str])
        assert_array_equal(y2, clf.predict(X2, lengths=[len(y), len(y)]))

    def test_perceptron_single_iter(self):
        """Assert that averaging works after a single iteration."""
        clf = StructuredPerceptron(max_iter=1)
        self.assertIsNotNone(clf.fit([[1, 2, 3]], [1], [1]))
