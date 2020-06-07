import unittest
from datetime import datetime

from airflow import DAG
from airflow.models import TaskInstance

from dsbox.dbconnection.plasma_connector import PlasmaConnector
from dsbox.operators.data_operator import DataOperator
from dsbox.operators.data_unit import DataInputFileUnit, DataOutputPlasmaUnit

from tests.config import socket_name, object_id


def drop_na_dataframe(dataframe, columns):
    dataframe = dataframe.dropna(subset=columns)
    return dataframe


class TestPlasmaConnector(unittest.TestCase):
    def test_execute_data_operator_csv_read_and_plasma_write(self):
        # given
        plasma_connector = PlasmaConnector(socket_name)

        dag = DAG(dag_id='test', start_date=datetime.now())
        input_csv_unit = DataInputFileUnit('data/X.csv', sep=';')
        output_plasma_unit = DataOutputPlasmaUnit(plasma_connector, object_id)

        task = DataOperator(operation_function=drop_na_dataframe,
                            params={'columns': ['ANNEEREALISATIONDIAGNOSTIC']},
                            input_unit=input_csv_unit,
                            output_unit=output_plasma_unit,
                            dag=dag, task_id='data_operator_csv_to_parquet')

        task_instance = TaskInstance(task=task, execution_date=datetime.now())

        # when
        task.execute(task_instance.get_template_context())

        # then
        other_plasma_connector = PlasmaConnector(socket_name)
        df_transformed = other_plasma_connector.get_dataframe(object_id)
        self.assertEqual((10245, 27), df_transformed.shape)


if __name__ == '__main__':
    unittest.main()
