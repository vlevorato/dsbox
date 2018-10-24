import unittest
from datetime import datetime

import pandas as pd
from airflow import DAG

from dsbox.operators.data_operator import DataOperator
from dsbox.operators.data_unit import DataInputFileUnit, DataOutputFileUnit
from dsbox.utils import execute_dag


def drop_na_dataframe(dataframe, columns):
    dataframe = dataframe.dropna(subset=columns)
    return dataframe


class TestDagFile(unittest.TestCase):
    def test_building_two_operators_with_execution(self):
        # given
        dag = DAG(dag_id='test_dag_file', start_date=datetime.now())

        input_csv_unit = DataInputFileUnit('data/X.csv', sep=';')
        output_parquet_unit = DataOutputFileUnit('data/X_parsed.parquet', pandas_write_function_name='to_parquet')
        task_1 = DataOperator(operation_function=drop_na_dataframe,
                              params={'columns': ['ANNEEREALISATIONDIAGNOSTIC']},
                              input_unit=input_csv_unit,
                              output_unit=output_parquet_unit,
                              dag=dag, task_id='data_operator_csv_to_parquet')

        input_parquet_unit = DataInputFileUnit('data/X_parsed.parquet', pandas_read_function_name='read_parquet')
        output_csv_unit = DataOutputFileUnit('data/X_parsed_2.csv', index=False)
        task_2 = DataOperator(operation_function=drop_na_dataframe,
                              params={'columns': ['ANNEETRAVAUXPRECONISESDIAG']},
                              input_unit=input_parquet_unit,
                              output_unit=output_csv_unit,
                              dag=dag, task_id='data_operator_parquet_to_csv')

        task_2.set_upstream(task_1)

        # when
        execute_dag(dag, verbose=True)

        # then
        df = pd.read_csv('data/X_parsed_2.csv')
        self.assertEqual((7241, 27), df.shape)
