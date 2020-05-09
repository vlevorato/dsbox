"""Hidden Markov models (HMMs) with supervised training."""

# Copyright 2013 Lars Buitinck

import numpy as np
from scipy.special import logsumexp

from dsbox.ml.markov.seqlearn.base import BaseSequenceClassifier
from dsbox.ml.markov.seqlearn._utils import atleast2d_or_csr, count_trans, safe_sparse_dot

__credits__ = "Lars Buitinck"

class MultinomialHMM(BaseSequenceClassifier):
    """First-order hidden Markov model with multinomial event model.

    Parameters
    ----------
    decode : string, optional
        Decoding algorithm, either "bestfirst" or "viterbi" (default).
        Best-first decoding is also called posterior decoding in the HMM
        literature.

    alpha : float
        Lidstone (additive) smoothing parameter.
        
    Examples
    --------
    
    >>> import pandas as pd
    >>> from dsbox.ml.markov.seqlearn import MultinomialHMM
    >>> clf = MultinomialHMM()
    >>> X_train = pd.DataFrame({'a': [0, 1, 2], 'b':[0, 1, 2], 'c': [0, 1, 2]})
    >>> y_train = np.array([0, 1, 2])
    >>> X_test = pd.DataFrame({'a': [1, 1, 1], 'b':[1, 1, 1], 'c': [1, 1, 1]})
    >>> _ = clf.fit(X_train, y_train, [len(y_train)])
    >>> clf.predict(X_test)
    array([2, 1, 2])
    """

    def __init__(self, decode="viterbi", alpha=.01):
        self.alpha = alpha
        self.decode = decode

    def fit(self, X, y, lengths):
        """Fit HMM model to data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Feature matrix of individual samples.

        y : array-like, shape (n_samples,)
            Target labels.

        lengths : array-like of integers, shape (n_sequences,)
            Lengths of the individual sequences in X, y. The sum of these
            should be n_samples.

        Notes
        -----
        Make sure the training set (X) is one-hot encoded; if more than one
        feature in X is on, the emission probabilities will be multiplied.

        Returns
        -------
        self : MultinomialHMM
        """

        alpha = self.alpha
        if alpha <= 0:
            raise ValueError("alpha should be >0, got {0!r}".format(alpha))

        X = atleast2d_or_csr(X)
        classes, y = np.unique(y, return_inverse=True)
        lengths = np.asarray(lengths)
        Y = y.reshape(-1, 1) == np.arange(len(classes))

        end = np.cumsum(lengths)
        start = end - lengths

        init_prob = np.log(Y[start].sum(axis=0) + alpha)
        init_prob -= logsumexp(init_prob)
        final_prob = np.log(Y[start].sum(axis=0) + alpha)
        final_prob -= logsumexp(final_prob)

        feature_prob = np.log(safe_sparse_dot(Y.T, X) + alpha)
        feature_prob -= logsumexp(feature_prob, axis=0)

        trans_prob = np.log(count_trans(y, len(classes)) + alpha)
        trans_prob -= logsumexp(trans_prob, axis=0)

        self.coef_ = feature_prob
        self.intercept_init_ = init_prob
        self.intercept_final_ = final_prob
        self.intercept_trans_ = trans_prob

        self.classes_ = classes

        return self
