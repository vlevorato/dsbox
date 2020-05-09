import matplotlib.pyplot as plt
import numpy as np

__author__ = "Samuel Rochette"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


def plot_eval_prediction_interval(target_eval, prediction, lower_error_list, upper_error_list, figsize=(10, 7)):
    """
    Display the sorted predictions of the evaluation set and their prediction intervals versus the ground truth.

    Parameters
    ----------
    target_eval : array-like
    prediction : array-like, prediction of the evaluation set
    lower_error_list : array-like
    upper_error_list : array-like
    figsize : tuple, optional (default=(10,7)), set the figure size

    Returns
    -------
    Prediction interval figure

    References
    ----------
    [1]  Quantile Regression Forests, Nicolai Meinshausen. 2006
    [2]  https://blog.datadive.net/prediction-intervals-for-random-forests/

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from dsbox.ml.explain.prediction_interval import compute_prediction_interval
    >>> from dsbox.ml.visualization.prediction_interval_visualization import plot_eval_prediction_interval

    >>> boston = load_boston()
    >>> data_train, data_eval, target_train, target_eval = train_test_split(boston['data'], boston['target'], test_size=0.3, random_state=42)
    
    >>> rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    >>> _ = rf.fit(data_train, target_train)
    
    >>> lower_error_list, upper_error_list = compute_prediction_interval(rf, data_eval, percentile=90)
    >>> prediction = rf.predict(data_eval)
    >>> plot_eval_prediction_interval(target_eval, prediction, lower_error_list, upper_error_list)
    """

    x = range(len(prediction))
    permutation_sorted_ground_truth = target_eval.argsort()
    y = target_eval[permutation_sorted_ground_truth]
    sorted_predictions = prediction[permutation_sorted_ground_truth]
    upper_error = np.array(upper_error_list)[permutation_sorted_ground_truth] - sorted_predictions
    lower_error = sorted_predictions - np.array(lower_error_list)[permutation_sorted_ground_truth]

    plt.figure(figsize=figsize)
    plt.plot(x, y, color='c', marker='o', markersize=6, label='True values', linestyle='None')
    plt.errorbar(x, sorted_predictions, yerr=[lower_error, upper_error], fmt='bo', label='Predicted values', capsize=3)
    plt.ylabel('Response value')
    plt.xlabel('Sample no.')
    plt.legend(loc='upper left')
    plt.show()


def plot_prediction_interval(prediction, lower_error_list, upper_error_list, figsize=(10, 7)):
    """
    Display the sorted predictions their prediction intervals.

    Parameters
    ----------
    prediction : array-like, prediction of the evaluation set
    lower_error_list : array-like
    upper_error_list : array-like
    figsize : tuple, optional (default=(10,7)), set the figure size

    Returns
    -------
    Prediction interval figure

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from dsbox.ml.explain.prediction_interval import compute_prediction_interval
    >>> from dsbox.ml.visualization.prediction_interval_visualization import plot_prediction_interval

    >>> boston = load_boston()
    >>> data_train, data_eval, target_train, target_eval = train_test_split(boston['data'], boston['target'], test_size=0.3, random_state=42)
    >>> rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    >>> _ = rf.fit(data_train, target_train)

    >>> lower_error_list, upper_error_list = compute_prediction_interval(rf, data_eval, percentile=90)
    >>> prediction = rf.predict(data_eval)

    >>> plot_prediction_interval(prediction, lower_error_list, upper_error_list)
    """

    x = range(len(prediction))
    permutation_sorted_prediction = prediction.argsort()
    sorted_predictions = prediction[permutation_sorted_prediction]
    upper_error = np.array(upper_error_list)[permutation_sorted_prediction] - sorted_predictions
    lower_error = sorted_predictions - np.array(lower_error_list)[permutation_sorted_prediction]

    plt.figure(figsize=figsize)
    plt.errorbar(x, sorted_predictions, yerr=[lower_error, upper_error], fmt='bo', label='Predicted values', capsize=3)
    plt.ylabel('Response value')
    plt.xlabel('Sample no.')
    plt.legend(loc='upper left')
    plt.show()
