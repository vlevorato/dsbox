import collections

import numpy as np
from sklearn.metrics import mean_squared_error, precision_recall_curve, recall_score, precision_score

__author__ = "Vincent Levorato"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


def generalized_jaccard_similarity_score(y_true, y_pred):
    """ Generalized Jaccard similarity score

    The Generalized Jaccard similarity [1] is an extension of the Jaccard similarity coefficient.
    It handles multisets as input variables.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float

    References
    ----------
    . [1] `Wikipedia entry for the Jaccard index
           <https://en.wikipedia.org/wiki/Jaccard_index#Generalized_Jaccard_similarity_and_distance>`_

    """

    min_count = 0
    max_count = 0

    counter_y_true = collections.Counter(y_true)
    counter_y_pred = collections.Counter(y_pred)

    # union of all elements from y_true and y_pred
    elements = set(y_true).union(set(y_pred))

    for element in elements:
        max_count += max(counter_y_true[element], counter_y_pred[element])
        min_count += min(counter_y_true[element], counter_y_pred[element])

    # could happen only if both collections are empty
    if max_count == 0:
        return 0.0

    return min_count / float(max_count)


def root_mean_squared_error(y_true, y_pred):
    """ Root Mean Square Error

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float

    References
    ----------
    . [1] `Wikipedia entry for the Root-mean-square deviation
           <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`

    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_squared_percentage_error(y_true, y_pred):
    """ Mean Square Percentage Error

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float

    References
    ----------
    . [1] `Wikipedia entry for the Mean percentage error
           <https://en.wikipedia.org/wiki/Mean_percentage_error>`

    """
    return np.mean(np.ma.masked_invalid(np.square((y_true - y_pred) / y_true)))


def root_mean_squared_percentage_error(y_true, y_pred):
    """ Root Mean Square Percentage Error

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float

    References
    ----------
    . [1] `Wikipedia entry for the Mean percentage error
           <https://en.wikipedia.org/wiki/Mean_percentage_error>`

    """
    return np.sqrt(mean_squared_percentage_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    """ Mean Absolute Percentage Error

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float

    References
    ----------
    . [1] `Wikipedia entry for the Mean absolute percentage error
           <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`

    """
    return np.mean(np.ma.masked_invalid(np.abs((y_true - y_pred) / y_true)))


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """ Symmetric Mean Absolute Percentage Error

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float
    """
    return np.mean(np.ma.masked_invalid(np.abs((y_true - y_pred) / ((y_true + y_pred) / 2))))


def symmetric_root_mean_squared_percentage_error(y_true, y_pred):
    """ Symetric Root Mean Square Percentage Error

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float
    """
    return np.sqrt(np.mean(np.ma.masked_invalid(np.square((y_true - y_pred) / ((y_true + y_pred) / 2)))))


def precision_with_fixed_recall(y_true, y_pred_proba, fixed_recall):
    """ Compute precision with a fixed recall, for class 1. The chosen threshold for this couple precision is also returned.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred_proba : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier, should be probabilities

    fixed_recall : float
        Fixed recall, recall will be calculated with respect to this precision

    Returns
    -------
    precision_score : float
    threshold : float
    """
    if is_valid_y_true(y_true):
        _, recall, threshold = precision_recall_curve(y_true, y_pred_proba)
        threshold = max([threshold[i] for i in range(len(threshold)) if recall[i] >= fixed_recall])
        y_pred_binary = binarize(y_pred_proba, threshold)
        return precision_score(y_true, y_pred_binary), threshold
    else:
        raise ValueError('y_true should not contain only zeros')


def recall_with_fixed_precision(y_true, y_pred_proba, fixed_precision):
    """ Compute recall with a fixed precision, for class 1. The chosen threshold for this couple recall is also returned.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred_proba : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier, should be probabilities

    fixed_precision : float
        Fixed precision, recall will be calculated with respect to this precision

    Returns
    -------
    recall_score : float
    threshold : float
    """
    if is_valid_y_true(y_true):
        precision, _, threshold = precision_recall_curve(y_true, y_pred_proba)
        threshold = min([threshold[i] for i in range(len(threshold)) if precision[i] >= fixed_precision])
        y_pred_binary = binarize(y_pred_proba, threshold)
        return recall_score(y_true, y_pred_binary), threshold
    else:
        raise ValueError('y_true should not contain only zeros')


def binarize(probas, threshold):
    """ Convert probabilities into 1s if superior or equal to the threshold, 0s otherwise.

    Parameters
    ----------
    probas : 1d array-like
        float probabilities to convert into binaries

    threshold : float

    Returns
    -------
    y_pred_binary : 1d array-like of 0s and 1s
    """
    return [1 if proba >= threshold else 0 for proba in probas]


def is_valid_y_true(y_true):
    """
    Return true if y_true contains at least one 1, false otherwise
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix

    Returns
    -------
    is_valid_y_true : bool
    """
    return np.sum(y_true) > 0
