import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from dsbox.utils import write_model_file, load_model_file


def join_dataframes(dataframe_list):
    X = dataframe_list[0]
    for dataframe in dataframe_list[1:len(dataframe_list)]:
        X = X.join(dataframe)

    return X


def dict_missing_feature(dataframe, missing_values_feature, groupby_feature, dummy_feature):
    df = dataframe.groupby([groupby_feature, missing_values_feature])[[dummy_feature]].count().reset_index(drop=False)
    df.columns = [groupby_feature, missing_values_feature, 'count']
    df_count = df.groupby(groupby_feature)[['count']].max().reset_index(drop=False)
    df = df.merge(df_count, on=[groupby_feature, 'count'])
    mapping_dict = df[[groupby_feature, missing_values_feature]].set_index(groupby_feature).to_dict()[
        missing_values_feature]

    return mapping_dict


def fillna_columns(dataframe, mode='train', model_path=None):
    X = dataframe

    if mode == 'train':
        imputer_ANNEEREALISATIONDIAGNOSTIC = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer_ANNEEREALISATIONDIAGNOSTIC.fit(X[['ANNEEREALISATIONDIAGNOSTIC']])
        write_model_file(model_path + 'imputer_ANNEEREALISATIONDIAGNOSTIC.feat', imputer_ANNEEREALISATIONDIAGNOSTIC)
    else:
        imputer_ANNEEREALISATIONDIAGNOSTIC = load_model_file(model_path + 'imputer_ANNEEREALISATIONDIAGNOSTIC.feat')

    X['ANNEEREALISATIONDIAGNOSTIC'] = imputer_ANNEEREALISATIONDIAGNOSTIC.transform(X[['ANNEEREALISATIONDIAGNOSTIC']])
    X['ANNEEREALISATIONDIAGNOSTIC'] = np.round(X['ANNEEREALISATIONDIAGNOSTIC']).astype('int')

    X['ANNEETRAVAUXPRECONISESDIAG'] = X['ANNEETRAVAUXPRECONISESDIAG'].fillna(-1)

    if mode == 'train':
        mapping_dict_espece = dict_missing_feature(X, 'ESPECE', 'GENRE_BOTA', 'CODE')
        write_model_file(model_path + 'mapping_dict_espece.feat', mapping_dict_espece)
    else:
        mapping_dict_espece = load_model_file(model_path + 'mapping_dict_espece.feat')

    genre_botas = X['GENRE_BOTA'].unique()
    for genre_bota in genre_botas:
        if genre_bota not in mapping_dict_espece:
            mapping_dict_espece[genre_bota] = 'inconnu'

    X['ESPECE'] = X.apply(lambda row: mapping_dict_espece[row['GENRE_BOTA']] if pd.isnull(row['ESPECE']) else row[
        'ESPECE'], axis=1)

    return X


def category_to_numerical_features(dataframe, mode='train', model_path=None):

