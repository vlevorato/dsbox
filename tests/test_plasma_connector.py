import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from dsbox.dbconnection.plasma_connector import PlasmaConnector
from tests.config import socket_name


class TestPlasmaConnector(unittest.TestCase):
    def test_put_and_get_dataframe_into_plasma_store(self):
        # given
        d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
             'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
        df = pd.DataFrame(d)

        plasma_store = PlasmaConnector(socket_name)

        # when
        df_id = plasma_store.put_dataframe(df)
        df_from_store = plasma_store.get_dataframe(df_id)

        # then
        assert_frame_equal(df, df_from_store)

        plasma_store.client.disconnect()
