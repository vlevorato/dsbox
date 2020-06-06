import unittest
from datetime import datetime
import pandas as pd

from airflow import DAG
from airflow.models import TaskInstance

from dsbox.operators.data_operator import DataOperator
from dsbox.operators.data_unit import DataInputFileUnit, DataOutputFileUnit


def drop_na_dataframe(dataframe, columns):
    dataframe = dataframe.dropna(subset=columns)
    return dataframe


class TestDataOperator(unittest.TestCase):
    def test_execute_data_operator_csv_read_and_write(self):
        # given
        dag = DAG(dag_id='test', start_date=datetime.now())
        input_csv_unit = DataInputFileUnit('data/X.csv', sep=';')
        output_csv_unit = DataOutputFileUnit('data/X_parsed.csv', index=False)

        task = DataOperator(operation_function=drop_na_dataframe,
                            params={'columns': ['ANNEEREALISATIONDIAGNOSTIC']},
                            input_unit=input_csv_unit,
                            output_unit=output_csv_unit,
                            dag=dag, task_id='data_operator_csv')
        task_instance = TaskInstance(task=task, execution_date=datetime.now())

        # when
        task.execute(None)

        # then
        df_transformed = pd.read_csv('data/X_parsed.csv')
        self.assertEqual((10245, 27), df_transformed.shape)

    def test_execute_data_operator_csv_read_and_parquet_write(self):
        # given
        dag = DAG(dag_id='test', start_date=datetime.now())
        input_csv_unit = DataInputFileUnit('data/X.csv', sep=';')
        output_parquet_unit = DataOutputFileUnit('data/X_parsed.parquet', pandas_write_function_name='to_parquet')

        task = DataOperator(operation_function=drop_na_dataframe,
                            params={'columns': ['ANNEEREALISATIONDIAGNOSTIC']},
                            input_unit=input_csv_unit,
                            output_unit=output_parquet_unit,
                            dag=dag, task_id='data_operator_csv_to_parquet')

        task_instance = TaskInstance(task=task, execution_date=datetime.now())

        # when
        task.execute(None)

        # then
        df_transformed = pd.read_parquet('data/X_parsed.parquet', engine='pyarrow')
        self.assertEqual((10245, 27), df_transformed.shape)


if __name__ == '__main__':
    unittest.main()
