from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from dsbox.utils import write_model_file, load_model_file


def fit_write_model(dataframe, columns_selection, column_target, write_path, model=DecisionTreeClassifier):
    X = dataframe[columns_selection]
    y = dataframe[column_target]

    clf = model()

    clf.fit(X, y)
    write_model_file(write_path, clf)


def read_predict_model(dataframe, columns_selection, read_path, y_pred_column_name='y_prediction'):
    X = dataframe[columns_selection]

    clf = load_model_file(read_path)
    y_pred = clf.predict(X)

    dataframe[y_pred_column_name] = y_pred
    return dataframe


def model_performance(dataframe, y_true_column_name, y_pred_column_name):
    print("F-score: " + str(f1_score(dataframe[y_true_column_name], dataframe[y_pred_column_name])))
