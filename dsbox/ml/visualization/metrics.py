import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_confusion_matrix(confusion_matrix, classes_list, normalize=True, figsize=(10, 7), fontsize=14, cmap="Blues"):
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
    >>> from dsbox.ml.visualization.metrics import print_confusion_matrix
    >>> array = [[ 8458,   227,  1730], \
             [ 1073, 37590,  1613], \
             [ 2390,  1159, 17540]]
    >>> classes_list = ["A", "B", "C"]
    >>> print_confusion_matrix(array, classes_list)
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
        plt.matshow(df_cm,  fignum=0, cmap=cmap)
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
