import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc

__author__ = "Aurélien Massiot"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


def plot_confusion_matrix(confusion_matrix, classes_list, normalize=True, figsize=(10, 7), fontsize=14, cmap="Blues"):
    """
    Display a pretty confusion matrix.

    Parameters
    ----------
    confusion_matrix : array-like

    classes_list : list,
        classes list of the confusion matrix

    normalize : boolean,
        normalize confusion matrix

    figsize : tuple, optional (default=(10,7))
        set the figure size

    fontsize : int, optional (default=14)
        set the font size

    cmap : str, optional (default="Blues")
        set the colormap

    Returns
    -------
    Confusion matrix figure


    Examples
    --------
    >>> from dsbox.ml.visualization.metrics import plot_confusion_matrix
    >>> array = [[ 8458,   227,  1730], \
             [ 1073, 37590,  1613], \
             [ 2390,  1159, 17540]]
    >>> classes_list = ["A", "B", "C"]
    >>> plot_confusion_matrix(array, classes_list)
   """
    confusion_matrix = np.array(confusion_matrix)

    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        normalized_cm = np.array(confusion_matrix).astype('float') / np.array(confusion_matrix).sum(axis=1)[:,
                                                                     np.newaxis]
        df_cm = pd.DataFrame(
            normalized_cm, index=classes_list, columns=classes_list,
        )
        plt.matshow(df_cm, fignum=0, cmap=cmap)
    else:
        df_cm = pd.DataFrame(
            confusion_matrix, index=classes_list, columns=classes_list,
        )
        plt.matshow(df_cm, fignum=0, cmap=cmap)
    ax.set_xticks(np.arange(len(classes_list)))
    ax.set_yticks(np.arange(len(classes_list)))
    ax.set_xticklabels(classes_list)
    ax.set_yticklabels(classes_list)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(classes_list)):
        for j in range(len(classes_list)):
            ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="grey", fontsize=fontsize)

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()


def plot_roc_curve(y_test, y_pred_probas, proba_step=None):
    """
    Plot ROC curve with probabilities thresholds.
    
    Parameters
    ----------
    y_test : array-like
        true labels
    
    y_pred_probas : array-like
        predicted labels
    
    proba_step : int (optional) (default=None)
        if set, give the step for each probability display. If None, nothing is displayed.

    Examples
    --------
    
    >>> from dsbox.ml.visualization.metrics import plot_roc_curve
    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    
    >>> X, y = datasets.make_moons(noise=0.3, random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
    
    >>> clf = RandomForestClassifier(n_estimators=10, random_state=42)
    >>> _ = clf.fit(X_train, y_train)
    >>> y_pred_probas = clf.predict_proba(X_test)
    
    >>> plot_roc_curve(y_test, y_pred_probas, proba_step=2)

    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probas[:, 1])
    auc_score = auc(fpr, tpr)

    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange', lw=lw, marker='.')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    if proba_step is not None:
        i = 0
        for x, y, txt in zip(fpr, tpr, thresholds):
            if i % proba_step == 0:
                plt.annotate(np.round(txt, 2), (x, y - 0.04), color='darkgray', fontsize=8)
            i += 1
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) - AUC score: {}'.format(str(np.round(auc_score,3))))
    plt.show()
