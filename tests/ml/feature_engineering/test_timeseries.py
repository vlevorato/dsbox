import unittest

import numpy as np
import pandas as pd

from dsbox.ml.feature_engineering.timeseries import Shifter, RollingWindower, DistributionTransformer, \
    np_rolling_agg_window, np_rolling_agg_abs_deviation_window
from dsbox.ml.outliers import median_absolute_deviation
from pandas.util.testing import assert_frame_equal, assert_numpy_array_equal, assert_series_equal


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


class TestRollingWindower(unittest.TestCase):
    def test_compute_moving_average(self):
        df = pd.DataFrame({'data': [0, 1, 2, 3, 4], 'data_bis': [3, 1, 0, 4, 1]})

        roller = RollingWindower(windows=[3], min_periods=1)

        # compute potential differences -> test will fail if a difference is present
        df_mean = pd.DataFrame({'mean_3_data': [0, 0.5, 1, 2, 3], 'mean_3_data_bis': [3, 2, 4 / 3.0, 5 / 3.0, 5 / 3.0]})
        df_test = df_mean.sub(roller.transform(df))
        self.assertEqual(df_test.sum().sum(), 0)

    def test_compute_custom_function(self):
        df = pd.DataFrame({'data': [0, 1, 2, 3, 4], 'data_bis': [3, 1, 0, 4, 1]})

        roller = RollingWindower(operation=median_absolute_deviation, windows=[3], min_periods=1)

        # compute potential differences -> test will fail if a difference is present
        df_mad_true = pd.DataFrame({'mad_3_data': [0, 0.5, 1, 1, 1], 'mad_3_data_bis': [0, 1, 1, 1, 1]})
        df_mad = roller.transform(df)
        df_test = df_mad_true.sub(df_mad)
        self.assertEqual(df_test.sum().sum(), 0)

    def test_compute_moving_average_with_datetime(self):
        df_ts = pd.DataFrame({'data': [0, 1, 2, 3, 4]},
                             index=[pd.Timestamp('20130101 09:00:00'),
                                    pd.Timestamp('20130101 09:00:02'),
                                    pd.Timestamp('20130101 09:00:03'),
                                    pd.Timestamp('20130101 09:00:05'),
                                    pd.Timestamp('20130101 09:00:06')])

        df_mean_ts = pd.DataFrame(
            {'mean_5s_data': [0, 0.5, 1, 2, 2.5],
             'mean_2s_data': [0, 1, 1.5, 3, 3.5]},
            index=[pd.Timestamp('20130101 09:00:00'),
                   pd.Timestamp('20130101 09:00:02'),
                   pd.Timestamp('20130101 09:00:03'),
                   pd.Timestamp('20130101 09:00:05'),
                   pd.Timestamp('20130101 09:00:06')]
        )

        roller = RollingWindower(windows=['5s', '2s'])

        df_test = df_mean_ts.sub(roller.transform(df_ts))
        self.assertEqual(df_test.sum().sum(), 0)


class TestDistribution(unittest.TestCase):
    def test_distribution_transformer_should_produce_df_distribution_per_column(self):
        # given
        df = pd.DataFrame({'sales': [3, 10, 12, 23, 48, 19, 21]})

        # when
        distrib_transformer = DistributionTransformer(3)
        df_distrib = distrib_transformer.fit_transform(df)

        # then
        df_expected = pd.DataFrame({'sales_bin_1': [0, 1, 1, 2, 1, 2, 2],
                                    'sales_bin_2': [0, 0, 0, 0, 1, 0, 0],
                                    'sales_bin_3': [1, 0, 0, 0, 0, 0, 0],
                                    'sales_bin_4': [0, 1, 2, 1, 1, 1, 1]})

        assert_frame_equal(df_expected, df_distrib)

    def test_distribution_transformer_should_produce_df_quantile_per_column(self):
        # given
        df = pd.DataFrame({'sales': [3, 10, 12, 23, 48, 19, 21]})

        # when
        distrib_transformer = DistributionTransformer(3, quantiles=[0., 0.25, 0.5, 0.75, 1.])
        df_distrib = distrib_transformer.fit_transform(df)

        # then
        df_expected = pd.DataFrame({'sales_quantile_1': [3., 3, 3, 10, 12, 19, 19],
                                    'sales_quantile_2': [3, 4.75, 6.5, 11, 17.5, 21, 20],
                                    'sales_quantile_3': [3, 6.5, 10, 12, 23, 23, 21],
                                    'sales_quantile_4': [3, 8.25, 11, 17.5, 35.5, 35.5, 34.5],
                                    'sales_quantile_5': [3., 10, 12, 23, 48, 48, 48],
                                    })

        assert_frame_equal(df_expected, df_distrib)


class TestNumpyRolling(unittest.TestCase):
    def test_np_rolling_mean_window_on_np_array(self):
        # given
        array = np.array([0, 1, 2, 3, 4])

        # when
        array_rol_mean = np_rolling_agg_window(array)

        # then
        array_expected = np.array([np.nan, np.nan, 1, 2, 3])

        assert_numpy_array_equal(array_expected, array_rol_mean)

    def test_np_rolling_median_window_on_np_array(self):
        # given
        array = np.array([0, 1, 2, 3, 0])

        # when
        array_rol_median = np_rolling_agg_window(array, agg_func=np.nanmedian)

        # then
        array_expected = np.array([np.nan, np.nan, 1, 2, 2])

        assert_numpy_array_equal(array_expected, array_rol_median)

    def test_np_rolling_mean_window_on_pd_dataframe(self):
        # given
        df = pd.DataFrame({'values': [0, 1, 2, 3, 4]})

        # when
        df_rol_mean = df.apply(np_rolling_agg_window)

        # then
        df_expected = pd.DataFrame({'values': [np.nan, np.nan, 1, 2, 3]})

        assert_frame_equal(df_expected, df_rol_mean)

    def test_np_rolling_mad_window_on_np_array(self):
        # given
        array = np.array([0, 1, 2, 3, 0])

        # when
        array_rol_mad = np.round(np_rolling_agg_abs_deviation_window(array), 2)

        # then
        array_expected = np.array([np.nan, np.nan, 0.67, 0.67, 1.11])

        assert_numpy_array_equal(array_expected, array_rol_mad)

    def test_np_rolling_mad_window_on_pd_dataframe(self):
        # given
        df = pd.DataFrame({'values': [0, 1, 2, 3, 0]})

        # when
        df_rol_mad = df.apply(np_rolling_agg_abs_deviation_window).round(2)

        # then
        df_expected = pd.DataFrame({'values': [np.nan, np.nan, 0.67, 0.67, 1.11]})

        assert_frame_equal(df_expected, df_rol_mad)

    def test_np_rolling_mad_window_on_pd_dataframe_using_groupby(self):
        # given
        df = pd.DataFrame({
            'group': ['A'] * 5 + ['B'] * 5,
            'values': [0, 1, 2, 3, 4, 0, 1, 2, 3, 0]})

        # when
        serie_rol_group_mad = df.groupby('group')['values'].apply(
            np_rolling_agg_abs_deviation_window).explode().reset_index(drop=True).apply(np.round, decimals=2)

        # then
        serie_expected = pd.Series(name='values',
                                   data=[np.nan, np.nan, 0.67, 0.67, 0.67, np.nan, np.nan, 0.67, 0.67, 1.11])

        assert_series_equal(serie_expected, serie_rol_group_mad)
