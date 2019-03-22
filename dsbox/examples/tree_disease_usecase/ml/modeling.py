from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from dsbox.utils import write_object_file, load_object_file


def generate_model():
    return RandomForestClassifier(n_estimators=100, n_jobs=-1)


def fit_write_model(dataframe, columns_selection, column_target, write_path, model=generate_model):
    X = dataframe[columns_selection]
    y = dataframe[column_target]

    clf = model()

    clf.fit(X, y)
    write_object_file(write_path, clf)


def read_predict_model(dataframe, columns_selection, read_path, y_pred_column_name='y_prediction'):
    X = dataframe[columns_selection]

    clf = load_object_file(read_path)
    y_pred = clf.predict(X)

    dataframe[y_pred_column_name] = y_pred
    return dataframe


def model_performance(dataframe, y_true_column_name, y_pred_column_name):
    print("Accuracy: " + str(accuracy_score(dataframe[y_true_column_name], dataframe[y_pred_column_name])))
    print("F-score: " + str(f1_score(dataframe[y_true_column_name], dataframe[y_pred_column_name])))
