import os
import unittest
from unittest import skipUnless

import pandas as pd
from pandas.testing import assert_frame_equal
from tests.config import test_user, test_dbname, test_hostname, test_password, test_port

from dsbox.dbconnection.dbconnector import DBconnector


@skipUnless(
    os.getenv('WITH_DB_TESTS'),
    'Please set environment variable WITH_DB_TESTS to enable these tests'
)
class TestDBConnector(unittest.TestCase):

    def test_write_df_to_table(self):
        # given
        df = pd.DataFrame({'town': ['Paris', 'New York', 'Roma'],
                           'country': ['France', 'USA', 'Italy'],
                           'trip_cost': [134.6, 234, 85.67]})

        dbconn = DBconnector(username=test_user, password=test_password, hostname=test_hostname, port=test_port,
                             dbname=test_dbname, baseprotocol='postgres://')

        dbconn.engine.execute('DROP TABLE IF EXISTS db.data_town')

        # when
        dbconn.df_to_table(df, 'data_town', schema='db', if_exists='replace', index=False)

        # then
        self.assertTrue(dbconn.check_table('data_town', schema='db'))

    def test_read_table_to_df(self):
        # given
        df_true = pd.DataFrame({'town': ['Paris', 'New York', 'Roma'],
                                'country': ['France', 'USA', 'Italy'],
                                'trip_cost': [134.6, 234, 85.67]})
        dbconn = DBconnector(username=test_user, password=test_password, hostname=test_hostname, port=test_port,
                             dbname=test_dbname, baseprotocol='postgres://')

        # when
        df_returned = dbconn.table_to_df('db.data_town')

        # then
        assert_frame_equal(df_true, df_returned)


if __name__ == '__main__':
    unittest.main()
