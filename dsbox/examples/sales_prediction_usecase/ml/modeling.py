import pandas as pd
import numpy as np
import lightgbm as lgb
from dsbox.ml.metrics import root_mean_squared_percentage_error
from dsbox.utils import write_object_file, load_object_file

lgbm_params = {
    'objective':'regression',
    'learning_rate':0.025,
    'n_estimators':1500,
    'max_depth':10
}


def extract_train_dataset(dataframe, min_date=20140101):
    X = dataframe
    X = X[X['Date'] >= min_date]
    X_for_train = X[X['to_predict'] == False]
    X_for_train = X_for_train[X_for_train['MissingValues'] == False]
    X_for_train = X_for_train[X_for_train['Open'] == 1]

    return X_for_train


def train_model(dataframe, features_to_exclude=[], target='Sales', model_path='', min_date=20140101):
    X_for_train = extract_train_dataset(dataframe, min_date=min_date)
    print(X_for_train.shape)

    cols_to_keep = list(set(X_for_train.columns) - set(features_to_exclude))
    print("Amount of features: " + str(len(cols_to_keep)))

    clf_gbm = lgb.LGBMRegressor(**lgbm_params)
    clf_gbm.fit(X_for_train[cols_to_keep], X_for_train[target])

    df_feat_importance = pd.DataFrame({'feature': cols_to_keep, 'importance': clf_gbm.feature_importances_})
    print(df_feat_importance.sort_values('importance', ascending=False))

    print("dumping model")
    write_object_file(model_path + 'lgbm.model', clf_gbm)


def metrics_model(dataframe, features_to_exclude=[], target='Sales', min_date=20140101, date_split=20150615):
    X_for_train = extract_train_dataset(dataframe, min_date=min_date)

    X_train = X_for_train[X_for_train['Date'] < date_split]
    X_test = X_for_train[X_for_train['Date'] >= date_split]

    cols_to_keep = list(set(X_for_train.columns) - set(features_to_exclude))
    print("Amount of features: " + str(len(cols_to_keep)))

    clf_gbm = lgb.LGBMRegressor(**lgbm_params)
    clf_gbm.fit(X_train[cols_to_keep], X_train[target],
                eval_set=[(X_test[cols_to_keep], X_test[target])],
                eval_metric='mse',
                early_stopping_rounds=20,
                verbose=1)

    df_feat_importance = pd.DataFrame({'feature': cols_to_keep, 'importance': clf_gbm.feature_importances_})
    print(df_feat_importance.sort_values('importance', ascending=False))

    y_pred_lgbm = clf_gbm.predict(X_test[cols_to_keep], num_iteration=clf_gbm.best_iteration_)

    score = root_mean_squared_percentage_error(X_test[target], y_pred_lgbm)
    print("RMSPE score: " + str(score))


def predict_model(dataframe, features_to_exclude=[], model_path=''):
    X = dataframe
    X_to_predict = X[X['to_predict'] == True]

    cols_to_keep = list(set(X.columns) - set(features_to_exclude))
    print("Amount of features: " + str(len(cols_to_keep)))

    clf_gbm = load_object_file(model_path + 'lgbm.model')
    y_pred = clf_gbm.predict(X_to_predict[cols_to_keep], num_iteration=clf_gbm.best_iteration_)

    X_to_predict['Sales_pred'] = np.round(y_pred).astype('int')
    X_to_predict['Open'] = X_to_predict['Open'].fillna(0)
    X_to_predict['Sales_pred'] = X_to_predict['Sales_pred'] * X_to_predict['Open']
    X_to_predict['Sales_pred'] = X_to_predict['Sales_pred'].map(lambda x: x if x >= 0 else 0)

    df_to_submit = X_to_predict[['Id', 'Sales_pred']].astype('int')
    df_to_submit['Sales'] = df_to_submit['Sales_pred']
    df_to_submit = df_to_submit.sort_values('Id')
    del df_to_submit['Sales_pred']

    return df_to_submit
