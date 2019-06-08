import numpy as np
import unittest

from scipy.special import logsumexp
from dsbox.ml.markov.hmmlearn.base import ConvergenceMonitor, _BaseHMM


class TestMonitor(unittest.TestCase):
    def test_converged_by_iterations(self):
        m = ConvergenceMonitor(tol=1e-3, n_iter=2, verbose=False)
        self.assertFalse(m.converged)
        m.report(-0.01)
        self.assertFalse(m.converged)
        m.report(-0.1)
        self.assertTrue(m.converged)

    def test_converged_by_logprob(self):
        m = ConvergenceMonitor(tol=1e-3, n_iter=10, verbose=False)
        for logprob in [-0.03, -0.02, -0.01]:
            m.report(logprob)
            self.assertFalse(m.converged)

        m.report(-0.0101)
        self.assertTrue(m.converged)

    def test_reset(self):
        m = ConvergenceMonitor(tol=1e-3, n_iter=10, verbose=False)
        m.iter = 1
        m.history.append(-0.01)
        m._reset()
        self.assertEqual(m.iter, 0)
        self.assertFalse(m.history)


class StubHMM(_BaseHMM):
    """An HMM with hardcoded observation probabilities."""

    def _compute_log_likelihood(self, X):
        return self.framelogprob


class TestBaseAgainstWikipedia(unittest.TestCase):
    def setup_method(self):
        # Example from http://en.wikipedia.org/wiki/Forward-backward_algorithm
        self.framelogprob = np.log([[0.9, 0.2],
                                    [0.9, 0.2],
                                    [0.1, 0.8],
                                    [0.9, 0.2],
                                    [0.9, 0.2]])

        h = StubHMM(2)
        h.transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        h.startprob_ = [0.5, 0.5]
        h.framelogprob = self.framelogprob
        self.hmm = h

    def test_do_forward_pass(self):
        self.setup_method()
        logprob, fwdlattice = self.hmm._do_forward_pass(self.framelogprob)

        reflogprob = -3.3725
        self.assertEqual(np.round(logprob, 4), reflogprob)
        reffwdlattice = np.array([[0.4500, 0.1000],
                                  [0.3105, 0.0410],
                                  [0.0230, 0.0975],
                                  [0.0408, 0.0150],
                                  [0.0298, 0.0046]])
        self.assertTrue(np.allclose(np.exp(fwdlattice), reffwdlattice, 4))

    def test_do_backward_pass(self):
        self.setup_method()
        bwdlattice = self.hmm._do_backward_pass(self.framelogprob)

        refbwdlattice = np.array([[0.0661, 0.0455],
                                  [0.0906, 0.1503],
                                  [0.4593, 0.2437],
                                  [0.6900, 0.4100],
                                  [1.0000, 1.0000]])
        self.assertTrue(np.allclose(np.exp(bwdlattice), refbwdlattice, 4))

    def test_do_viterbi_pass(self):
        self.setup_method()
        logprob, state_sequence = self.hmm._do_viterbi_pass(self.framelogprob)

        refstate_sequence = [0, 0, 1, 0, 0]
        self.assertTrue(np.allclose(state_sequence, refstate_sequence))

        reflogprob = -4.4590
        self.assertEqual(np.round(logprob, 4), reflogprob)

    def test_score_samples(self):
        self.setup_method()
        # ``StubHMM` ignores the values in ```X``, so we just pass in an
        # array of the appropriate shape.
        logprob, posteriors = self.hmm.score_samples(self.framelogprob)
        self.assertTrue(np.allclose(posteriors.sum(axis=1), np.ones(len(posteriors))))

        reflogprob = -3.3725
        self.assertEqual(np.round(logprob, 4), reflogprob)

        refposteriors = np.array([[0.8673, 0.1327],
                                  [0.8204, 0.1796],
                                  [0.3075, 0.6925],
                                  [0.8204, 0.1796],
                                  [0.8673, 0.1327]])
        self.assertTrue(np.allclose(posteriors, refposteriors, atol=1e-4))


class TestBaseConsistentWithGMM(unittest.TestCase):
    def setup_method(self):
        self.n_components = 8
        n_samples = 10

        self.framelogprob = np.log(np.random.random((n_samples, self.n_components)))

        h = StubHMM(self.n_components)
        h.framelogprob = self.framelogprob

        # If startprob and transmat are uniform across all states (the
        # default), the transitions are uninformative - the model
        # reduces to a GMM with uniform mixing weights (in terms of
        # posteriors, not likelihoods).
        h.startprob_ = np.ones(self.n_components) / self.n_components
        h.transmat_ = np.ones((self.n_components, self.n_components)) / self.n_components

        self.hmm = h

    def test_score_samples(self):
        self.setup_method()
        logprob, hmmposteriors = self.hmm.score_samples(self.framelogprob)

        n_samples, n_components = self.framelogprob.shape
        self.assertTrue(np.allclose(hmmposteriors.sum(axis=1), np.ones(n_samples)))

        norm = logsumexp(self.framelogprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(self.framelogprob - np.tile(norm, (1, self.n_components)))
        self.assertTrue(np.allclose(hmmposteriors, gmmposteriors))

    def test_decode(self):
        self.setup_method()
        _logprob, state_sequence = self.hmm.decode(self.framelogprob)

        n_samples, n_components = self.framelogprob.shape
        norm = logsumexp(self.framelogprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(self.framelogprob -
                               np.tile(norm, (1, n_components)))
        gmmstate_sequence = gmmposteriors.argmax(axis=1)
        self.assertTrue(np.allclose(state_sequence, gmmstate_sequence))


class TestHMMAttributes(unittest.TestCase):
    def test_base_hmm_attributes(self):
        n_components = 20
        startprob = np.random.random(n_components)
        startprob /= startprob.sum()
        transmat = np.random.random((n_components, n_components))
        transmat /= np.tile(transmat.sum(axis=1)[:, np.newaxis], (1, n_components))

        h = StubHMM(n_components)

        self.assertEqual(h.n_components, n_components)

        h.startprob_ = startprob
        self.assertTrue(np.allclose(h.startprob_, startprob))

        with self.assertRaises(ValueError):
            h.startprob_ = 2 * startprob
            h._check()
        with self.assertRaises(ValueError):
            h.startprob_ = []
            h._check()
        with self.assertRaises(ValueError):
            h.startprob_ = np.zeros((n_components - 2, 2))
            h._check()

        h.startprob_ = startprob
        h.transmat_ = transmat
        self.assertTrue(np.allclose(h.transmat_, transmat))
        with self.assertRaises(ValueError):
            h.transmat_ = 2 * transmat
            h._check()
        with self.assertRaises(ValueError):
            h.transmat_ = []
            h._check()
        with self.assertRaises(ValueError):
            h.transmat_ = np.zeros((n_components - 2, n_components))
            h._check()


class TestStationaryDistribution(unittest.TestCase):
    def test_stationary_distribution(self):
        n_components = 10
        h = StubHMM(n_components)
        transmat = np.random.random((n_components, n_components))
        transmat /= np.tile(transmat.sum(axis=1)[:, np.newaxis], (1, n_components))
        h.transmat_ = transmat
        stationary = h.get_stationary_distribution()
        self.assertEqual(stationary.dtype, float)
        self.assertListEqual(list(np.round(np.dot(h.get_stationary_distribution().T, h.transmat_), 5)),
                             list(np.round(stationary, 5)))
