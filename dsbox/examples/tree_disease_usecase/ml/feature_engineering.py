import numpy as np
from sklearn.impute import SimpleImputer

from dsbox.utils import write_model_file, load_model_file


def join_dataframes(dataframe_list):
    X = dataframe_list[0]
    for dataframe in dataframe_list[1:len(dataframe_list)]:
        X = X.join(dataframe)

    return X


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

    return X
