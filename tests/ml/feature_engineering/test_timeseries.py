import unittest

import numpy as np
import pandas as pd

from dsbox.ml.feature_engineering.timeseries import Shifter


class TestComputeShift(unittest.TestCase):
    def test_shift_ts(self):
        # given
        df_ts = pd.DataFrame({'data': [0.0, 1.0, 2.0, 3.0, 4.0]},
                             index=[pd.Timestamp('20130101 09:00:00'),
                                    pd.Timestamp('20130101 09:00:02'),
                                    pd.Timestamp('20130101 09:00:03'),
                                    pd.Timestamp('20130101 09:00:05'),
                                    pd.Timestamp('20130101 09:00:06')])

        # when
        shifter = Shifter(shifts=[1, 2], prefix='t-')
        df_shifted = shifter.transform(df_ts)

        # then
        df_expected = pd.DataFrame(
            {'t-1_data': [np.nan, 0, 1.0, 2.0, 3.0],
             't-2_data': [np.nan, np.nan, 0.0, 1.0, 2.0]},
            index=[pd.Timestamp('20130101 09:00:00'),
                   pd.Timestamp('20130101 09:00:02'),
                   pd.Timestamp('20130101 09:00:03'),
                   pd.Timestamp('20130101 09:00:05'),
                   pd.Timestamp('20130101 09:00:06')]
        )

        self.assertTrue(df_expected.equals(df_shifted))

    def test_shift_several_features(self):
        # given
        df = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0, 4.0],
                           'y': [0.0, 0.0, 2.0, 2.0, 1.0]})

        # when
        shifter = Shifter(shifts=[1, 2])
        df_shifted = shifter.transform(df)

        # then
        df_expected = pd.DataFrame(
            {'x_1': [np.nan, 0.0, 1.0, 2.0, 3.0],
             'x_2': [np.nan, np.nan, 0.0, 1.0, 2.0],
             'y_1': [np.nan, 0.0, 0.0, 2.0, 2.0],
             'y_2': [np.nan, np.nan, 0.0, 0.0, 2.0],
             }
        )

        self.assertTrue(df_expected.equals(df_shifted[df_expected.columns]))

    def test_shift_groupby(self):
        # given
        df = pd.DataFrame({'data': [0.0, 1.0, 2.0, 3.0, 4.0],
                           'category': ['A', 'A', 'A', 'B', 'B']})

        # when
        shifter = Shifter(shifts=[1, 2])
        df_shifted = shifter.transform(df.groupby('category'))

        # then
        df_expected = pd.DataFrame(
            {'data_1': [np.nan, 0.0, 1.0, np.nan, 3.0],
             'data_2': [np.nan, np.nan, 0.0, np.nan, np.nan]
             }
        )

        self.assertTrue(df_expected.equals(df_shifted[df_expected.columns]))

    def test_shift_empty_df(self):
        # given
        df = pd.DataFrame()

        # when
        shifter = Shifter()
        df_shifted = shifter.transform(df)

        # then
        df_expected = pd.DataFrame()

        self.assertTrue(df_expected.equals(df_shifted[df_expected.columns]))
