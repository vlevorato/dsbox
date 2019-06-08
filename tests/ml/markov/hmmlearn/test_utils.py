import unittest
import numpy as np

from dsbox.ml.markov.hmmlearn import normalize, fill_covars


class TestUtils(unittest.TestCase):
    def test_normalize(self):
        A = np.random.normal(42., size=128)
        A[np.random.choice(len(A), size=16)] = 0.0
        self.assertTrue((A == 0.0).any())
        normalize(A)
        self.assertTrue(np.allclose(A.sum(), 1.))

    def test_normalize_along_axis(self):
        A = np.random.normal(42., size=(128, 4))
        for axis in range(A.ndim):
            A[np.random.choice(len(A), size=16), axis] = 0.0
            self.assertTrue((A[:, axis] == 0.0).any())
            normalize(A, axis=axis)
            self.assertTrue(np.allclose(A.sum(axis=axis), 1.))

    def test_fill_covars(self):
        full = np.arange(12).reshape(3, 2, 2) + 1
        np.testing.assert_equal(fill_covars(full, 'full', 3, 2), full)

        diag = np.arange(6).reshape(3, 2) + 1
        expected = np.array([[[1, 0], [0, 2]],
                             [[3, 0], [0, 4]],
                             [[5, 0], [0, 6]]])
        np.testing.assert_equal(fill_covars(diag, 'diag', 3, 2), expected)

        tied = np.arange(4).reshape(2, 2) + 1
        expected = np.array([[[1, 2], [3, 4]],
                             [[1, 2], [3, 4]],
                             [[1, 2], [3, 4]]])
        np.testing.assert_equal(fill_covars(tied, 'tied', 3, 2), expected)

        spherical = np.array([1, 2, 3])
        expected = np.array([[[1, 0], [0, 1]],
                             [[2, 0], [0, 2]],
                             [[3, 0], [0, 3]]])
        np.testing.assert_equal(
            fill_covars(spherical, 'spherical', 3, 2), expected)
