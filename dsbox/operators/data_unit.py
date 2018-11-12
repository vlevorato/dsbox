from abc import ABC, abstractmethod

import pandas as pd


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
        print(input_path)

    def read_data(self):
        dataframe = getattr(pd, self.pandas_read_function_name)(self.input_path, **self.pandas_kwargs_read)
        return dataframe


class DataOutputFileUnit(DataOutputUnit):
    def __init__(self, output_path, pandas_write_function_name='to_csv', **kwargs):
        self.output_path = output_path
        self.pandas_write_function_name = pandas_write_function_name
        self.pandas_kwargs_write = kwargs
        print(output_path)

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
