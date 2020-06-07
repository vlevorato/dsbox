import io

import pandas as pd
import psycopg2 as pg
import sqlalchemy


class DBconnector():
    """
    Simple DB connector to write pandas dataframe to RDBMS table or to read from a table to a pandas dataframe.
    """

    def __init__(self, username, password, hostname, port, dbname, baseprotocol=''):
        self.username = username
        self.password = password
        self.hostname = hostname
        self.port = port
        self.dbname = dbname
        self.basetype = baseprotocol

        self.uri = baseprotocol + self.username + ':' + self.password + '@' + self.hostname + '/' + self.dbname
        self.engine = sqlalchemy.create_engine(self.uri)

    def df_to_table(self, df, table_name, **kwargs):
        """
        Write pandas dataframe to table.

        Parameters
        ----------
        df: pandas.DataFrame
            dataframe to export
        table_name: str
            target table name
        """
        df.to_sql(table_name, self.engine, **kwargs)

    def table_to_df(self, table_name, **kwargs):
        """
        Returns pandas dataframe from table.

        Parameters
        ----------
        table_name: str
            Table to read

        Returns
        -------
        pandas.DataFrame
            Result dataframe.

        """
        return pd.read_sql_query('SELECT * from ' + table_name, self.engine, **kwargs)

    def check_table(self, table_name, schema=None):
        """
        Check if a table exists.

        Parameters
        ----------
        table_name: str
            table name to check
        schema: str, default=None
            schema where lays the table to check

        Returns
        -------
        bool:
             Returns True if table exists, else returns False.
        """
        if schema is None:
            return self.engine.has_table(table_name)
        else:
            return self.engine.has_table(table_name, schema=schema)


class DBconnectorPG(DBconnector):
    """
        Specialized Postgres DB connector to write pandas dataframe to RDBMS table or to read from a table to a pandas dataframe.
        Using bulk operations.
    """

    def __init__(self, username, password, hostname, port, dbname):
        super(DBconnectorPG, self).__init__(username, password, hostname, port, dbname, baseprotocol='postgres://')

        self.con = pg.connect(user=self.username, password=self.password,
                              host=self.hostname, port=self.port, dbname=self.dbname)
        self.con.autocommit = True

    def bulk_to_pg(self, df, table_name, to_pg_drop=False):
        """
        Writes a dataframe into a PG table with bulk operations.

        Parameters
        ----------
        df: pandas.DataFrame
            dataframe to write
        table_name: str
            target table name
        to_pg_drop: bool, default=False
            specify if target table must be overwritten
        """

        data = io.StringIO()
        df.to_csv(data, header=True, index=False, sep=',', quotechar='"')
        data.seek(0)
        cursor = self.con.cursor()
        if to_pg_drop:
            cursor.execute("DROP TABLE IF EXISTS " + table_name)
            empty_table = pd.io.sql.get_schema(df, table_name, con=self.engine)
            empty_table = empty_table.replace('"', '')
            cursor.execute(empty_table)
        sql_query = "COPY " + table_name + " FROM STDIN WITH DELIMITER ',' CSV HEADER QUOTE '" + '"' + "'"
        cursor.copy_expert(sql_query, data)
        cursor.connection.commit()

    def bulk_from_pg(self, table_name):
        """
        Read a PG table to dataframe with bulk operations.

        Parameters
        ----------
        table_name: str
            table to read

        Returns
        -------
        df: pandas.DataFrame
            dataframe result

        """
        buf = io.StringIO()
        cursor = self.con.cursor()
        sql_query = "COPY " + table_name + " TO STDOUT WITH DELIMITER ',' CSV HEADER QUOTE '" + '"' + "' FORCE QUOTE *"
        cursor.copy_expert(sql_query, buf)
        buf.seek(0)
        df = pd.read_csv(buf)
        buf.close()

        return df

    def check_table_pg(self, schema_name, table_name):
        """
        Check if a PG table exists.

        Parameters
        ----------
        schema_name: str
            schema where lays the table to check

        table_name: str
            table name to check

        Returns
        -------
        bool:
             Returns True if table exists, else returns False.
        """
        query = "SELECT 1" \
                " FROM information_schema.tables" \
                " WHERE table_schema = '" + schema_name + "'" + \
                " AND table_name = '" + table_name + "'"

        df = pd.read_sql_query(query, self.engine)
        if len(df) > 0:
            return True
        else:
            return False
