def join_dataframes(dataframe_list):
    X = dataframe_list[0]
    for dataframe in dataframe_list[1:len(dataframe_list)]:
        X = X.join(dataframe)

    return X
