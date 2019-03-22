import pandas as pd

from dsbox.ml.feature_engineering import TagEncoder
from dsbox.ml.feature_engineering.timeseries import RollingWindower, Shifter
from dsbox.utils import pandas_downcast_numeric


def concat_train_test(dataframe_list):
    shop_data = dataframe_list[0]
    shop_data_to_predict = dataframe_list[1]
    shop_data['to_predict'] = False
    shop_data_to_predict['to_predict'] = True
    shop_data = pd.concat([shop_data, shop_data_to_predict], sort=False)
    shop_data = shop_data.reset_index(drop=True)
    return shop_data


def resample_fillna(dataframe):
    shop_data = dataframe
    shop_data['Datetime'] = pd.to_datetime(shop_data['Date'], format='%Y-%m-%d')
    shop_data = shop_data.set_index('Datetime')
    print("Resampling...")
    shop_data = shop_data.groupby('Store').resample('D').asfreq()
    shop_data = shop_data.drop('Store', axis=1)
    shop_data = shop_data.reset_index(0)

    print("Fill missing values...")
    shop_data['MissingValues'] = shop_data['Sales'].isnull()

    shop_data['DayOfWeek'] = shop_data.index.map(lambda x: x.dayofweek)
    shop_data['Month'] = shop_data.index.map(lambda x: x.month)
    sales_per_store_month_dayofweek = shop_data.groupby(['Store', 'Month', 'DayOfWeek'])['Sales'].median()
    customers_per_store_month_dayofweek = shop_data.groupby(['Store', 'Month', 'DayOfWeek'])['Customers'].median()
    dict_sales_per_store_month_dayofweek = sales_per_store_month_dayofweek.to_dict()
    dict_customers_per_store_month_dayofweek = customers_per_store_month_dayofweek.to_dict()

    shop_data['T_StoreMonthDow'] = list(zip(shop_data['Store'], shop_data['Month'], shop_data['DayOfWeek']))
    shop_data['SalesNew'] = shop_data['T_StoreMonthDow'].map(lambda x: dict_sales_per_store_month_dayofweek[x])
    shop_data['CustomersNew'] = shop_data['T_StoreMonthDow'].map(lambda x: dict_customers_per_store_month_dayofweek[x])
    shop_data['SalesNew'] = shop_data['SalesNew'] * shop_data['MissingValues']
    shop_data['CustomersNew'] = shop_data['CustomersNew'] * shop_data['MissingValues']
    shop_data['Sales'] = shop_data['Sales'].fillna(0) + shop_data['SalesNew']
    shop_data['Customers'] = shop_data['Customers'].fillna(0) + shop_data['CustomersNew']
    shop_data = shop_data.drop(['SalesNew', 'CustomersNew', 'T_StoreMonthDow'], axis=1)

    shop_data = shop_data.reset_index(drop=False)

    pandas_downcast_numeric(shop_data)

    print("Remaining missing values:\n" + str(shop_data[['Sales', 'Customers']].isnull().sum()))

    return shop_data


def date_to_dateint(date_val):
    return date_val.year * 10000 + date_val.month * 100 + date_val.day


def create_simple_features(dataframe):
    X = dataframe

    X['Date'] = X['Datetime'].apply(lambda x: date_to_dateint(x))
    X['Week'] = X['Datetime'].apply(lambda x: x.week)
    X['Year'] = X['Datetime'].apply(lambda x: x.year)

    X['SalePerCustomer'] = X['Sales'] / X['Customers']
    X['SalePerCustomer'] = X['SalePerCustomer'].fillna(0)

    col_list = ['StateHoliday']
    for col in col_list:
        te = TagEncoder()
        X[col] = te.fit_transform(X[col].fillna('nan'))

    pandas_downcast_numeric(X)

    return X


def create_timeseries_rolling_features(X_ts, operation='mean', features=['Sales'], shift_ranges=[42],
                                       rolling_windows=[7]):
    print(str(operation))
    X_ts = X_ts.sort_values(['Store', 'Date'])
    roller = RollingWindower(operation=operation, windows=rolling_windows, min_periods=1)
    X_roll = roller.transform(X_ts.groupby('Store')[features]).reset_index(0, drop=False)
    roll_shifter = Shifter(shifts=shift_ranges, prefix='d-')
    X_roll_shifted = roll_shifter.transform(X_roll.groupby('Store'))
    X_ts = X_ts.join(X_roll_shifted)
    X_ts = X_ts.drop(features, axis=1)

    return X_ts


def create_timeseries_shift_features(X_ts, features=['Sales'], shift_ranges=[42]):
    print("shift features")
    X_ts = X_ts.sort_values(['Store', 'Date'])
    shifter = Shifter(shifts=shift_ranges, prefix='d-')
    X_shifted = shifter.transform(X_ts.groupby('Store')[features])
    X_ts = X_ts.join(X_shifted)
    X_ts = X_ts.drop(features, axis=1)

    return X_ts


def create_timeseries_diff_shift_features(df_shift, features=['Sales'], shift_ranges=[42, 47], prefix='diff_'):
    for feat in features:
        for i in range(0, len(shift_ranges) - 1):
            df_shift[prefix + 'd-' + str(shift_ranges[i + 1]) + '_' + str(shift_ranges[i]) + '_' + feat] = \
                df_shift['d-' + str(shift_ranges[i + 1]) + '_' + feat] - df_shift[
                    'd-' + str(shift_ranges[i]) + '_' + feat]

    return df_shift


def merge_time_features(dataframe_list, features=['Sales'], merge_cols=['Store', 'Date']):
    for i in range(0, len(dataframe_list)):
        dataframe_list[i] = dataframe_list[i].sort_values(merge_cols).reset_index(drop=True)

    print("Merge...")
    passed_features = dataframe_list[0]
    for i in range(1, len(dataframe_list)):
        print('df ' + str(i))
        for col in dataframe_list[i].columns:
            if col not in merge_cols:
                print(col)
                passed_features[col] = dataframe_list[i][col]

    print(passed_features.columns)

    passed_features = passed_features.fillna(method='backfill')
    passed_features = passed_features.fillna(method='ffill')

    return passed_features


def extra_features(dataframe):
    extra_data = dataframe
    cols_to_fill = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek',
                    'Promo2SinceYear']
    for col in cols_to_fill:
        extra_data[col] = extra_data[col].fillna(extra_data[col].median())

    extra_data = extra_data.fillna(0)
    for col in ['StoreType', 'Assortment', 'PromoInterval']:
        te = TagEncoder()
        extra_data[col] = te.fit_transform(extra_data[col].fillna('nan'))

    for col in cols_to_fill:
        extra_data[col] = extra_data[col].astype('int')

    pandas_downcast_numeric(extra_data)

    return extra_data


def merge_final_features(dataframe_list):
    X = dataframe_list[0].merge(dataframe_list[1], on='Store')

    X['CompetitionOpen'] = 12 * (X['Year'] - X['CompetitionOpenSinceYear']) + \
                           (X['Month'] - X['CompetitionOpenSinceMonth'])
    X['PromoOpen'] = 12 * (X['Year'] - X['Promo2SinceYear']) + \
                     (X['Week'] - X['Promo2SinceWeek']) / 4.0

    return X