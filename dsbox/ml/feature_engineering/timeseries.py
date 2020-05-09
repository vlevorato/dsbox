import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import types

__author__ = "Vincent Levorato"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


class Shifter(BaseEstimator, TransformerMixin):
    """
        Compute shifted values for a given dataframe, and creates columns associated.

        Parameters
        ----------   

            shifts: list, optional, default [1]
                Allows to shift data by periods (meaning by row) using the pandas.shift() method.

            prefix: string, optional
                Put a prefix in front of the column names.

            suffix: string, optional
                Put a suffix at the end of the column names.

            group_by_cols: list, optional, default None
                Allows to shift values grouped by columns.

            ignore_cols: list, optional, default None
                Allows to ignore some columns to be not shifted.

        Examples
        --------

        >>> from dsbox.ml.feature_engineering.timeseries import Shifter
        >>> df_ts = pd.DataFrame({'data': [0.0, 1.0, 2.0, 3.0, 4.0]}, \
                             index=[pd.Timestamp('20130101 09:00:00'), \
                                    pd.Timestamp('20130101 09:00:02'), \
                                    pd.Timestamp('20130101 09:00:03'), \
                                    pd.Timestamp('20130101 09:00:05'), \
                                    pd.Timestamp('20130101 09:00:06')])
        >>> shifter = Shifter(shifts=[1],prefix='t-')
        >>> df_shift_ts = shifter.transform(df_ts)
        >>> df_shift_ts    
                             t-1_data
        2013-01-01 09:00:00       NaN
        2013-01-01 09:00:02       0.0
        2013-01-01 09:00:03       1.0
        2013-01-01 09:00:05       2.0
        2013-01-01 09:00:06       3.0


    """

    def __init__(self, shifts=[1], prefix='', suffix='', **kwargs):
        self.shifts = shifts
        self.prefix = prefix
        self.suffix = suffix
        self.kwargs = kwargs

    def fit(self, X=None, y=None):
        """
        No-op.
        This method doesn't do anything. It exists purely for compatibility
        with the scikit-learn transformer API.

        Parameters
        ----------
            X: array-like
            y: array-like

        Returns
        -------
            self: Shifter

        """

        return self

    def transform(self, X):
        """
        Transform a dataframe into shift values corresponding to shift periods

        Parameters
        ----------
            X: dataframe
                Input pandas dataframe.

        Returns
        -------
            X: dataframe 
                Dataframe with columns having shifted values (one by shift)

        """

        X_shifted = pd.DataFrame()

        for shift_value in self.shifts:
            X_shifted_tmp = X
            X_shifted_tmp = X_shifted_tmp.shift(periods=shift_value, **self.kwargs)

            prefix = ''
            suffix = ''
            if self.prefix != '':
                prefix = self.prefix + str(shift_value) + '_'
            if self.suffix != '':
                suffix = self.suffix + str(shift_value)
            if self.suffix == '' and self.prefix == '':
                suffix = '_' + str(shift_value)

            X_shifted_tmp.columns = X_shifted_tmp.columns.map(lambda x: prefix + x + suffix)
            if len(X_shifted) == 0:
                X_shifted = X_shifted_tmp
            else:
                X_shifted = X_shifted.join(X_shifted_tmp)

        return X_shifted


class RollingWindower(BaseEstimator, TransformerMixin):
    """
    Compute rolling aggregated values for a given dataframe.

    Classical operators (mean, std, median, etc.) or custom operators can be used to compute the windows. 
    Windows are based on index, which could be a simple integer or a pandas timestamp. Use **kwargs to pass extra
    arguments to pandas rolling function (like min_periods for instance).


    Parameters
    ----------   
        operation: str or function, optional, default 'mean'
            Set the aggregation function used to aggregate a window.

        windows: list, optional, default [3]
            Set the windows used to aggregate data. Time windows can be set also (see pandas time unit
            syntax) if the dataframe index is a timestamp
            Examples:
                [3]: one window of size 3
                [2,5]: one window of size 2 and one window of size 5
                [2s, 10s]: one window of 2 secs and one window of 10 secs

        diff_mode: boolean, optional, default False
            Process the difference between values and its window aggregate value.


    Examples
    --------
        >>> import pandas as pd
        >>> from dsbox.ml.feature_engineering.timeseries import RollingWindower
        >>> df = pd.DataFrame({'data': [0, 1, 2, 3, 4]})
        >>> roller = RollingWindower(windows=[2,3])
        >>> df_roll = roller.transform(df)
        >>> df_roll
           mean_2_data  mean_3_data
        0          NaN          NaN
        1          0.5          NaN
        2          1.5          1.0
        3          2.5          2.0
        4          3.5          3.0

        >>> df_ts = pd.DataFrame({'data': [0, 1, 2, 3, 4]}, \
                            index=[pd.Timestamp('20130101 09:00:00'), \
                                pd.Timestamp('20130101 09:00:02'), \
                                pd.Timestamp('20130101 09:00:03'), \
                                pd.Timestamp('20130101 09:00:05'), \
                                pd.Timestamp('20130101 09:00:06')])
        >>> roller = RollingWindower(windows=['5s', '2s'])
        >>> df_roll = roller.transform(df_ts)
        >>> df_roll
                             mean_5s_data  mean_2s_data
        2013-01-01 09:00:00           0.0           0.0
        2013-01-01 09:00:02           0.5           1.0
        2013-01-01 09:00:03           1.0           1.5
        2013-01-01 09:00:05           2.0           3.0
        2013-01-01 09:00:06           2.5           3.5



    """

    def __init__(self, operation='mean', windows=[3], **kwargs):
        self.operation = operation
        self.windows = windows
        self.kwargs = kwargs

    def fit(self, X=None, y=None):
        """
        No-op.
        This method doesn't do anything. It exists purely for compatibility
        with the scikit-learn transformer API.

        Parameters
        ----------
            X: array-like
            y: array-like

        Returns
        -------
            self: RollingWindower

        """

        return self

    def transform(self, raw_X):
        """
        Transform a dataframe into aggregated values corresponding to window sizes

        Parameters
        ----------
            raw_X: dataframe
                Input pandas dataframe.

        Returns
        -------
            X: dataframe 
                Dataframe with columns having rolling measures (one per window)

        """

        X = pd.DataFrame()

        for window in self.windows:

            X_m = raw_X.rolling(window, **self.kwargs).agg(self.operation)

            columns_name = []

            if isinstance(self.operation, types.FunctionType):
                col_name = self.operation.__name__
            else:
                col_name = self.operation

            for col in X_m.columns:
                columns_name.append(col_name + '_' + str(window) + '_' + col)

            X_m.columns = columns_name

            if len(X) == 0:
                X = X_m
            else:
                X = X.join(X_m)

        return X


class DistributionTransformer(BaseEstimator, TransformerMixin):
    """
    Build a discrete distribution (histogram) for feature engineering for each column, per line, 
    following a rolling window. It captures the evolving distribution of a feature.

    For instance, if a serie is composed of the following values: [3, 10, 12, 23], and with a window parameter of 3,
    it takes these values, and apply histogram (bins=4) function on it:
    [3] => [0, 0, 1, 0]
    [3, 10] => [1, 0, 0, 1]
    [3, 10, 12] => [1, 0, 0, 2]
    [10, 12, 23] => [2, 0, 0, 1]

    Parameters
    ----------   
        window: int
            Size of the rolling window.

        bins: int, optional, (default=4)
            Amount of bins used to estimate distribution
        
        quantiles: int, optional, (default=None)
            If set, the transformer will return quantiles information.

    Examples
    --------
    >>> import pandas as pd
    >>> from dsbox.ml.feature_engineering.timeseries import DistributionTransformer

    >>> df = pd.DataFrame({'sales': [3, 10, 12, 23, 48, 19, 21]})
    >>> distrib_transformer = DistributionTransformer(3)
    >>> distrib_transformer.fit_transform(df)
       sales_bin_1  sales_bin_2  sales_bin_3  sales_bin_4
    0            0            0            1            0
    1            1            0            0            1
    2            1            0            0            2
    3            2            0            0            1
    4            1            1            0            1
    5            2            0            0            1
    6            2            0            0            1
    """

    def __init__(self, window, bins=4, quantiles=None):
        self.window = window
        self.bins = bins
        self.quantiles = quantiles

    def fit(self, X=None, y=None):
        """
        No-op.
        This method doesn't do anything. It exists purely for compatibility
        with the scikit-learn transformer API.

        Parameters
        ----------
            X: array-like
            y: array-like

        Returns
        -------
            self

        """

        return self

    def transform(self, X):
        """
        Transform a dataframe to build discrete distribution per column.

        Parameters
        ----------
           X: dataframe
               Input pandas dataframe.

        Returns
        -------
           X: dataframe 
               Dataframe with bin values per column.

        """

        X_distrib = pd.DataFrame()

        for col in X.columns:
            col_serie = X[col]
            bins_list = []

            for i in range(0, len(col_serie)):
                min_bound = i - self.window + 1
                if min_bound < 0:
                    min_bound = 0
                max_bound = i + 1
                if max_bound >= len(col_serie):
                    max_bound = len(col_serie)
                if self.quantiles is None:
                    bins_list.append(np.histogram(col_serie[min_bound:max_bound], bins=self.bins)[0])
                else:
                    bins_list.append(np.quantile(col_serie[min_bound:max_bound], self.quantiles))

            X_col_distrib = pd.DataFrame(bins_list)
            X_col_distrib = X_col_distrib.set_index(X.index)
            if self.quantiles is None:
                X_col_distrib.columns = [col + '_bin_' + str(i) for i in range(1, self.bins + 1)]
            else:
                X_col_distrib.columns = [col + '_quantile_' + str(i) for i in range(1, len(self.quantiles) + 1)]
                X_col_distrib = X_col_distrib.fillna(0)

            if len(X_distrib) == 0:
                X_distrib = X_col_distrib
            else:
                X_distrib = X_distrib.join(X_col_distrib)

        return X_distrib


def transform_datetxt2int(X, col_date, format='%Y-%m-%d'):
    """
    Inplace transformation of a string date column into an integer format date column.

    Parameters
    ----------
    X: dataframe

    col_date: str
        Column name to transform

    format: str
        Pandas date str format

    """

    X[col_date] = pd.to_datetime(X[col_date], format=format)
    X[col_date] = X[col_date].map(lambda x: (x.year * 10 ** 4) + (x.month * 10 ** 2) + x.day)
    X[col_date] = X[col_date].astype('int')


def create_diff_shift_features(df_shift, cols=[], prefix='diff_'):
    """
    Create diff values between time series columns with have been shifted.

    Parameters
    ----------
    df_shift: dataframe
        Dataset with columns to make apply diff.
    cols: list
        Columns names, in order, to apply diff.
    prefix: str
        Prefix used to name diff columns.

    """
    for i in range(0, len(cols) - 1):
        df_shift[prefix + cols[i + 1] + '_' + cols[i]] = df_shift[cols[i + 1]] - df_shift[cols[i]]