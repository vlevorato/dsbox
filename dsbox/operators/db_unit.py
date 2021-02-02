from sqlalchemy import create_engine
import pandas as pd

from dsbox.operators.data_unit import DataOutputUnit, DataInputUnit
from dsbox.dbconnection.dbconnector import DBconnectorPG


class DataInputDBUnit(DataInputUnit):
    """
    Data unit allowing to read a DB table, with SQL engine creation on the fly. It uses Pandas read_sql
    function.

    Parameters
    ----------
    sql_query: str
        SQL query to read data
    db_url: str
        database url
    kwargs: dict
        used by Pandas API to pass others parameters
    """

    def __init__(self, sql_query, db_url, **kwargs):
        self.sql_query = sql_query
        self.db_url = db_url
        self.pandas_kwargs_read = kwargs

    def read_data(self):
        engine = create_engine(self.db_url, echo=False)
        return pd.read_sql(self.sql_query, engine, **self.pandas_kwargs_read)

    def __str__(self):
        return "query: \n{}".format(self.sql_query)


class DataOutputDBUnit(DataOutputUnit):
    """
    Data unit allowing to write a dataframe to a DB table, with SQL engine creation on the fly.
    It uses Pandas to_sql function.

    Parameters
    ----------
    output_name: str
        target SQL table
    db_url: str
        database url
    kwargs: dict
        used by Pandas API to pass others parameters
    """

    def __init__(self, output_name, db_url, **kwargs):
        self.output_name = output_name
        self.db_url = db_url
        self.pandas_kwargs_write = kwargs

    def write_data(self, dataframe):
        engine = create_engine(self.db_url, echo=False)
        dataframe.to_sql(self.output_name, engine, **self.pandas_kwargs_write)

    def __str__(self):
        return self.output_name


class DataPGUnit:
    """
    Global data unit class specilized in PG database operations.

    connection_infos_dict: dict
        dict containing all connection information: username, password, hostname, port, dbname

    """

    def __init__(self, connection_infos_dict):
        self.username = connection_infos_dict['username']
        self.password = connection_infos_dict['password']
        self.hostname = connection_infos_dict['hostname']
        self.port = connection_infos_dict['port']
        self.dbname = connection_infos_dict['dbname']


class DataInputPGUnit(DataPGUnit, DataInputUnit):
    """
    Data unit allowing to read a PG DB table, using efficient bulk operations.

    connection_infos_dict: dict
        dict containing all connection information: username, password, hostname, port, dbname
    """

    def __init__(self, connection_infos_dict, **kwargs):
        super(DataInputPGUnit, self).__init__(connection_infos_dict)
        self.dbconnector_kwargs = kwargs

    def read_data(self):
        dbconnector = DBconnectorPG(self.username, self.password, self.hostname, self.port, self.dbname)
        return dbconnector.bulk_from_pg(**self.dbconnector_kwargs)

    def __str__(self):
        return self.dbconnector_kwargs['table_name']


class DataOutputPGUnit(DataPGUnit, DataOutputUnit):
    """
    Data unit allowing to write a PG DB table, using efficient bulk operations.

    connection_infos_dict: dict
        dict containing all connection information: username, password, hostname, port, dbname
    """

    def __init__(self, connection_infos_dict, **kwargs):
        super(DataOutputPGUnit, self).__init__(connection_infos_dict)
        self.dbconnector_kwargs = kwargs

    def write_data(self, dataframe):
        dbconnector = DBconnectorPG(self.username, self.password, self.hostname, self.port, self.dbname)
        dbconnector.bulk_to_pg(dataframe, **self.dbconnector_kwargs)

    def __str__(self):
        return self.dbconnector_kwargs['table_name']
