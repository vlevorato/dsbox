import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from pyproj import Proj, transform

from dsbox.ml.feature_engineering import TagEncoder
from dsbox.utils import write_object_file, load_object_file


def lambert_coord_to_long_lat(x, y, inProj, outProj):
    return transform(inProj, outProj, x, y)


def transform_df_coordinates(dataframe, x_column, y_column):
    inProj = Proj(init="epsg:3945")
    outProj = Proj(proj='lonlat', ellps='WGS84')

    dataframe[['long', 'lat']] = dataframe.apply(lambda row: lambert_coord_to_long_lat(row[x_column], row[y_column],
                                                                                       inProj, outProj), axis=1) \
        .apply(pd.Series)

    return dataframe


def join_dataframes(dataframe_list, **kwargs):
    X = dataframe_list[0]
    for dataframe in dataframe_list[1:len(dataframe_list)]:
        X = X.join(dataframe, **kwargs)

    return X


def dict_missing_feature(dataframe, missing_values_feature, groupby_feature, dummy_feature):
    df = dataframe.groupby([groupby_feature, missing_values_feature])[[dummy_feature]].count().reset_index(drop=False)
    df.columns = [groupby_feature, missing_values_feature, 'count']
    df_count = df.groupby(groupby_feature)[['count']].max().reset_index(drop=False)
    df = df.merge(df_count, on=[groupby_feature, 'count'])
    mapping_dict = df[[groupby_feature, missing_values_feature]].set_index(groupby_feature).to_dict()[
        missing_values_feature]

    return mapping_dict


def notediag_to_num(text):
    if text == "Arbre d'avenir normal":
        return 0
    if text == "Arbre d'avenir incertain":
        return 1
    if "dans les 10 ans" in text:
        return 2
    if "dans les 5 ans" in text:
        return 3
    if "diatement" in text:
        return 4

    return -1


def prio_renouv_to_num(text):
    if text == "plus de 20 ans":
        return 20
    if "de 11" in text:
        return 11
    if "de 6" in text:
        return 6
    if "de 1 " in text:
        return 1

    return -1


def remarques_to_num(text):
    if pd.isnull(text) or text == '0':
        return 0

    if "mort" in text:
        return 2

    return 1


def fillna_columns(dataframe, simple_features=[], mode='train', model_path=None):
    X = dataframe

    if mode == 'train':
        imputer_ANNEEREALISATIONDIAGNOSTIC = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer_ANNEEREALISATIONDIAGNOSTIC.fit(X[['ANNEEREALISATIONDIAGNOSTIC']])
        write_object_file(model_path + 'imputer_ANNEEREALISATIONDIAGNOSTIC.feat', imputer_ANNEEREALISATIONDIAGNOSTIC)
    else:
        imputer_ANNEEREALISATIONDIAGNOSTIC = load_object_file(model_path + 'imputer_ANNEEREALISATIONDIAGNOSTIC.feat')

    X['ANNEEREALISATIONDIAGNOSTIC'] = imputer_ANNEEREALISATIONDIAGNOSTIC.transform(X[['ANNEEREALISATIONDIAGNOSTIC']])
    X['ANNEEREALISATIONDIAGNOSTIC'] = np.round(X['ANNEEREALISATIONDIAGNOSTIC']).astype('int')

    X['ANNEETRAVAUXPRECONISESDIAG'] = X['ANNEETRAVAUXPRECONISESDIAG'].fillna(-1)

    miss_group_features = [('ESPECE', 'GENRE_BOTA'),
                           ('DIAMETREARBREAUNMETRE', 'ESPECE'),
                           ('VARIETE', 'ESPECE')
                           ]
    for tuple in miss_group_features:

        missing_feature = tuple[0]
        grouping_feature = tuple[1]

        if mode == 'train':
            mapping_dict = dict_missing_feature(X, missing_feature, grouping_feature, 'CODE')
            write_object_file(model_path + 'mapping_dict_' + missing_feature + '.feat', mapping_dict)
        else:
            mapping_dict = load_object_file(model_path + 'mapping_dict_' + missing_feature + '.feat')

        grouping_feature_values = X[grouping_feature].unique()
        for grouping_feature_value in grouping_feature_values:
            if grouping_feature_value not in mapping_dict:
                mapping_dict[grouping_feature_value] = 'inconnu'

        X[missing_feature] = X.apply(
            lambda row: mapping_dict[row[grouping_feature]] if pd.isnull(row[missing_feature]) else row[
                missing_feature], axis=1)

    for feature in simple_features:
        X[feature] = X[feature].fillna('inconnu')

    return X


def category_to_numerical_features(dataframe, features, mode='train', model_path=None):
    X = dataframe

    for feature in features:
        if mode == 'train':
            tagencoder = TagEncoder()
            X[feature] = tagencoder.fit_transform(X[feature])
            write_object_file(model_path + 'tagencoder_' + feature + '.feat', tagencoder)
        else:
            tagencoder = load_object_file(model_path + 'tagencoder_' + feature + '.feat')
            X[feature] = tagencoder.transform(X[feature])

    X['DIAMETREARBREAUNMETRE'] = X['DIAMETREARBREAUNMETRE'].map(lambda x: '-1' if x == 'inconnu' else x)
    X['DIAMETREARBREAUNMETRE'] = X['DIAMETREARBREAUNMETRE'].map(lambda x: x.split(' ')[0]).astype('int')

    X['NOTEDIAGNOSTIC'] = X['NOTEDIAGNOSTIC'].apply(notediag_to_num)
    X['PRIORITEDERENOUVELLEMENT'] = X['PRIORITEDERENOUVELLEMENT'].apply(prio_renouv_to_num)
    X['REMARQUES'] = X['REMARQUES'].apply(remarques_to_num)

    return X
