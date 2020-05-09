import random
import numpy as np

__author__ = "Vincent Levorato"
__license__ = "Apache 2.0"


def greedy_feature_selection(X_train, X_test, y_train, y_test, model, columns, score_func, shuffle=False):
    """
    Greedy technique to select features in a model. It takes all the columns passed into the function at first, and remove
    feature one by one, recalculating performance each time (train/test/score). If removing a feature increases the score
    it is evinced, else, it is kept.

    Parameters
    ----------
        X_train: pandas.Dataframe
        X_test: pandas.Dataframe
        y_train : array-like
        y_test : array-like
        model: sklearn estimator
        columns : array-like
        score_func : function
        shuffle : bool (optional)

    Returns
    -------
    Return a sublist of features

    """
    cols_selected = list(columns)

    if shuffle:
        random.shuffle(cols_selected)

    i = 0
    stop = False

    model.fit(X_train[cols_selected], y_train)
    y_pred = np.round(model.predict(X_test[cols_selected]))
    score = score_func(y_test, y_pred)
    print(score)
    print('')

    while not stop:
        col_to_remove = cols_selected.pop(0)
        print(col_to_remove)
        model.fit(X_train[cols_selected], y_train)
        y_pred = np.round(model.predict(X_test[cols_selected]))

        score_temp = score_func(y_test, y_pred)
        if score_temp <= score:
            score = score_temp
            print(score)
            i = 0
        else:
            cols_selected.append(col_to_remove)

        i += 1

        if i == len(cols_selected):
            stop = True
            print("No more optim")

    return cols_selected
