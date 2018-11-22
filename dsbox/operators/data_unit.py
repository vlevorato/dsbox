from abc import ABC, abstractmethod

import pandas as pd
from sqlalchemy import create_engine

from dsbox.dbconnection.dbconnector import DBconnectorPG


class DataInputUnit(ABC):
    @abstractmethod
    def read_data(self):
        pass


class DataOutputUnit(ABC):
    @abstractmethod
    def write_data(self, dataframe):
        pass


class DataInputFileUnit(DataInputUnit):
    def __init__(self, input_path, pandas_read_function_name='read_csv', **kwargs):
        self.input_path = input_path
        self.pandas_read_function_name = pandas_read_function_name
        self.pandas_kwargs_read = kwargs

    def read_data(self):
        dataframe = getattr(pd, self.pandas_read_function_name)(self.input_path, **self.pandas_kwargs_read)
        return dataframe


class DataOutputFileUnit(DataOutputUnit):
    def __init__(self, output_path, pandas_write_function_name='to_csv', **kwargs):
        self.output_path = output_path
        self.pandas_write_function_name = pandas_write_function_name
        self.pandas_kwargs_write = kwargs

    def write_data(self, dataframe):
        getattr(dataframe, self.pandas_write_function_name)(self.output_path, **self.pandas_kwargs_write)


class DataInputPlasmaUnit(DataInputUnit):
    def __init__(self, plasma_store, object_id):
        self.plasma_store = plasma_store
        self.object_id = object_id

    def read_data(self):
        return self.plasma_store.get_dataframe(self.object_id)


class DataOutputPlasmaUnit(DataOutputUnit):
    def __init__(self, plasma_store, object_id, overwrite=True):
        self.plasma_store = plasma_store
        self.object_id = object_id
        self.overwrite = overwrite

    def write_data(self, dataframe):
        self.plasma_store.put_dataframe(dataframe, self.object_id, overwrite=self.overwrite)


class DataInputMultiFileUnit(DataInputUnit):
    def __init__(self, input_path_list, pandas_read_function_name='read_csv', **kwargs):
        self.input_path_list = input_path_list
        self.pandas_read_function_name = pandas_read_function_name
        self.pandas_kwargs_read = kwargs

    def read_data(self):
        dataframe_list = []
        for input_path in self.input_path_list:
            dataframe_list.append(getattr(pd, self.pandas_read_function_name)(input_path, **self.pandas_kwargs_read))
        return dataframe_list

class DataInputDBUnit(DataInputUnit):
    def __init__(self, sql_query, db_url,  **kwargs):
        self.sql_query = sql_query
        self.db_url = db_url
        self.pandas_kwargs_read = kwargs

    def read_data(self):
        engine = create_engine(self.db_url, echo=False)
        return pd.read_sql(self.sql_query, engine, **self.pandas_kwargs_read)

class DataOutputDBUnit(DataOutputUnit):
    def __init__(self, output_name, db_url,  **kwargs):
        self.output_name = output_name
        self.db_url = db_url
        self.pandas_kwargs_write = kwargs

    def write_data(self, dataframe):
        engine = create_engine(self.db_url, echo=False)
        dataframe.to_sql(self.output_name, engine, **self.pandas_kwargs_write)

class DataPGUnit:
    def __init__(self, connection_infos_dict):
        self.username = connection_infos_dict['username']
        self.password = connection_infos_dict['password']
        self.hostname = connection_infos_dict['hostname']
        self.port = connection_infos_dict['port']
        self.dbname = connection_infos_dict['dbname']


class DataInputPGUnit(DataPGUnit, DataInputUnit):
    def __init__(self, connection_infos_dict, **kwargs):
        super(DataInputPGUnit, self).__init__(connection_infos_dict)
        self.dbconnector_kwargs = kwargs

    def read_data(self):
        dbconnector = DBconnectorPG(self.username, self.password, self.hostname, self.port, self.dbname)
        return dbconnector.bulk_from_pg(**self.dbconnector_kwargs)

class DataOutputPGUnit(DataPGUnit, DataOutputUnit):
    def __init__(self, connection_infos_dict, **kwargs):
        super(DataOutputPGUnit, self).__init__(connection_infos_dict)
        self.dbconnector_kwargs = kwargs

    def write_data(self, dataframe):
        dbconnector = DBconnectorPG(self.username, self.password, self.hostname, self.port, self.dbname)
        dbconnector.bulk_to_pg(dataframe, **self.dbconnector_kwargs)