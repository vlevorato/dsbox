import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
