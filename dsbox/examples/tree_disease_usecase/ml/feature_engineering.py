import numpy as np

def join_dataframes(dataframe_list):
    X = dataframe_list[0]
    for dataframe in dataframe_list[1:len(dataframe_list)]:
        X = X.join(dataframe)

    return X


def fillna_columns(dataframe):
    X = dataframe

    mean_feat = np.round(X['ANNEEREALISATIONDIAGNOSTIC'].mean()).astype('int')
    X['ANNEEREALISATIONDIAGNOSTIC'] = X['ANNEEREALISATIONDIAGNOSTIC'].fillna(mean_feat)

    X['ANNEETRAVAUXPRECONISESDIAG'] = X['ANNEETRAVAUXPRECONISESDIAG'].fillna(-1)

    return X