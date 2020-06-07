from abc import ABC, abstractmethod
from importlib import import_module

import pandas as pd
from sqlalchemy import create_engine

from dsbox.dbconnection.dbconnector import DBconnectorPG


class DataInputUnit(ABC):
    """
    Abstract class defining reading contract for all Data Unit with input data.
    """
    input_path = None

    @abstractmethod
    def read_data(self):
        pass

    def __str__(self):
        return str(self.input_path)


class DataOutputUnit(ABC):
    """
    Abstract class defining reading contract for all Data Unit with output data.
    """
    output_path = None

    @abstractmethod
    def write_data(self, dataframe):
        pass

    def __str__(self):
        return str(self.output_path)


class DataGlobalInputUnit(DataInputUnit):
    """
    Data unit allowing to read data via any API using a dataframe structure (Pandas, Dask, Vaex).

    Parameters
    ----------
    input_path: str
        file path to read
    api_module: str, default='pandas'
        module name used apply read function set in read_function parameter
    read_function_name: str, default='read_csv'
        set the function name used by backend API to read data
    kwargs: dict
        used by backend API to pass others parameters
    """

    def __init__(self, input_path, api_module='pandas', read_function_name='read_csv', **kwargs):
        self.input_path = input_path
        self.read_function_name = read_function_name
        self.api_module = import_module(api_module)
        self.kwargs_read = kwargs

    def read_data(self):
        dataframe = getattr(self.api_module, self.read_function_name)(self.input_path, **self.kwargs_read)
        return dataframe


class DataGlobalOutputFileUnit(DataOutputUnit):
    """
    Data unit allowing to write data via dataframe API.

    Parameters
    ----------
    output_path: str
        file path to write
    write_function_name: str, default='to_csv'
        set the function name used by backend API to write data
    kwargs: dict
        used by backend API to pass others parameters
    """

    def __init__(self, output_path, write_function_name='to_csv', **kwargs):
        self.output_path = output_path
        self.write_function_name = write_function_name
        self.kwargs_write = kwargs

    def write_data(self, dataframe):
        getattr(dataframe, self.write_function_name)(self.output_path, **self.kwargs_write)


class DataInputFileUnit(DataInputUnit):
    """
    Data unit allowing to read data via Pandas API.

    Parameters
    ----------
    input_path: str
        file path to read
    pandas_read_function_name: str, default='read_csv'
        set the function name used by Pandas to read data
    kwargs: dict
        used by Pandas API to pass others parameters
    """

    def __init__(self, input_path, pandas_read_function_name='read_csv', **kwargs):
        self.input_path = input_path
        self.pandas_read_function_name = pandas_read_function_name
        self.pandas_kwargs_read = kwargs

    def read_data(self):
        dataframe = getattr(pd, self.pandas_read_function_name)(self.input_path, **self.pandas_kwargs_read)
        return dataframe


class DataInputPathUnit(DataInputUnit):
    """
    Lazy version of the DataInputFileUnit class, passing only the path to read_data method,
    not the loaded dataframe.

    Parameters
    ----------
    input_path: str
        file path to read
    """

    def __init__(self, input_path):
        self.input_path = input_path

    def read_data(self):
        return self.input_path


class DataOutputFileUnit(DataGlobalOutputFileUnit):
    """
    Data unit allowing to write data via Pandas API. (kept for backwards compatibility)

    Parameters
    ----------
    output_path: str
        file path to write
    pandas_write_function_name: str, default='to_csv'
        set the function name used by Pandas to write data
    kwargs: dict
        used by Pandas API to pass others parameters
    """

    def __init__(self, output_path, pandas_write_function_name='to_csv', **kwargs):
        super(DataOutputFileUnit, self).__init__(output_path, pandas_write_function_name, **kwargs)

    def write_data(self, dataframe):
        getattr(dataframe, self.write_function_name)(self.output_path, **self.kwargs_write)


class DataInputPlasmaUnit(DataInputUnit):
    """
    Data unit allowing to read persisted Pandas dataframe in memory using Arrow Plasma store.

    Parameters
    ----------
    plasma_store: PlasmaConnector
        object used to connect to Plasma store
    object_id: int
        unique id used to read dataframe
    """

    def __init__(self, plasma_store, object_id):
        self.plasma_store = plasma_store
        self.object_id = object_id

    def read_data(self):
        return self.plasma_store.get_dataframe(self.object_id)

    def __str__(self):
        return "{}/{}".format(self.plasma_store, str(self.object_id))


class DataOutputPlasmaUnit(DataOutputUnit):
    """
    Data unit allowing to persist Pandas dataframe in memory using Arrow Plasma store.

    Parameters
    ----------
    plasma_store: PlasmaConnector
        object used to connect to Plasma store
    object_id: int
        unique id used to write dataframe
    overwrite: bool, default=True
        if True, don't verify if object id is still present, and overwrite data, else, an exception is raised
    """

    def __init__(self, plasma_store, object_id, overwrite=True):
        self.plasma_store = plasma_store
        self.object_id = object_id
        self.overwrite = overwrite

    def write_data(self, dataframe):
        self.plasma_store.put_dataframe(dataframe, self.object_id, overwrite=self.overwrite)

    def __str__(self):
        return "{}/{}".format(self.plasma_store, str(self.object_id))


class DataInputMultiFileUnit(DataInputUnit):
    """
    Data unit allowing to read several data sources at once via Pandas API. The read_data method returns
    a list of loaded dataframes.
    Note: data format has to be the same for all data sources.

    Parameters
    ----------
    input_path_list: list
        list of paths data to read
    pandas_read_function_name: str, default='read_csv'
        set the function name used by Pandas to read data
    kwargs: dict
        used by Pandas API to pass others parameters
    """

    def __init__(self, input_path_list, pandas_read_function_name='read_csv', **kwargs):
        self.input_path_list = input_path_list
        self.pandas_read_function_name = pandas_read_function_name
        self.pandas_kwargs_read = kwargs

    def read_data(self):
        dataframe_list = []
        for input_path in self.input_path_list:
            dataframe_list.append(getattr(pd, self.pandas_read_function_name)(input_path, **self.pandas_kwargs_read))
        return dataframe_list

    def __str__(self):
        return str(self.input_path_list)


class DataInputMultiPathUnit(DataInputUnit):
    """
    Lazy version of the DataInputMultiFileUnit class, passing only the path to read_data method,
    not the loaded dataframe.

    Parameters
    ----------
    input_path_list: list
        list of paths data to read
    """

    def __init__(self, input_path_list):
        self.input_path_list = input_path_list

    def read_data(self):
        return self.input_path_list

    def __str__(self):
        return str(self.input_path_list)


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
