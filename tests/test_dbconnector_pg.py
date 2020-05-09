import os
import unittest
from unittest import skipUnless

import pandas as pd
from pandas.testing import assert_frame_equal
from tests.config import test_user, test_dbname, test_hostname, test_password, test_port

from dsbox.dbconnection.dbconnector import DBconnectorPG


@skipUnless(
    os.getenv('WITH_DB_TESTS'),
    'Please set environment variable WITH_DB_TESTS to enable these tests'
)
class TestDBConnectorPG(unittest.TestCase):

    def test_write_df_to_table_with_to_pg_parameter(self):
        # given
        df = pd.DataFrame({'town': ['Paris', 'New York', 'Roma'],
                           'country': ['France', 'USA', 'Italy'],
                           'trip_cost': [134.6, 234, 85.67]})

        dbconn = DBconnectorPG(username=test_user, password=test_password, hostname=test_hostname, port=test_port,
                               dbname=test_dbname)

        # when
        dbconn.bulk_to_pg(df, 'db.data_town', to_pg_drop=True)

        # then
        self.assertTrue(dbconn.check_table_pg('db', 'data_town'))

    def test_read_table_to_df_from_pg(self):
        # given
        df_true = pd.DataFrame({'town': ['Paris', 'New York', 'Roma'],
                                'country': ['France', 'USA', 'Italy'],
                                'trip_cost': [134.6, 234, 85.67]})
        dbconn = DBconnectorPG(username=test_user, password=test_password, hostname=test_hostname, port=test_port,
                               dbname=test_dbname)

        # when
        df_returned = dbconn.bulk_from_pg('db.data_town')

        # then
        assert_frame_equal(df_true, df_returned)


if __name__ == '__main__':
    unittest.main()
