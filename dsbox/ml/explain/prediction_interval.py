import numpy as np

__author__ = "Samuel Rochette"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


def compute_prediction_interval(model, data_eval, percentile=95):
    """
    A function that compute the prediction interval based on the predictions of each trees of a random forest.

        Parameters
        ----------
        model: a fitted RandomForestRegressor.

        data_eval: A panda or numpy matrix of data of the validation set.

        percentile: Percentile in the interval. Default value 95.

        Returns
        -------
        Two arrays : lower_bound_prediction_list and upper_bound_prediction_list

        References
        ----------
        [1]  Quantile Regression Forests, Nicolai Meinshausen. 2006
        [2]  https://blog.datadive.net/prediction-intervals-for-random-forests/
    """

    decision_tree_list = model.estimators_
    lower_bound = (100 - percentile) / 2.
    upper_bound = 100 - lower_bound

    # this is the matrix of each predictions for each trees
    decision_tree_prediction_matrix = [_compute_prediction_of_each_tree(decision_tree_list, data_eval[i]) for i in
                                       range(len(data_eval))]

    lower_bound_prediction_list = [np.percentile(prediction_list, lower_bound) for prediction_list in
                                   decision_tree_prediction_matrix]
    upper_bound_prediction_list = [np.percentile(prediction_list, upper_bound) for prediction_list in
                                   decision_tree_prediction_matrix]

    return lower_bound_prediction_list, upper_bound_prediction_list


def _compute_prediction_of_each_tree(decision_tree_list, row):
    # reshaping is mandatory to avoid scikit warning
    reshaped_row = row.reshape(1, -1)
    return [decision_tree.predict(reshaped_row)[0] for decision_tree in decision_tree_list]
