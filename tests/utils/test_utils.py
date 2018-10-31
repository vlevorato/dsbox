import unittest
from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator

from dsbox.operators.data_operator import DataOperator
from dsbox.operators.data_unit import DataInputFileUnit, DataOutputFileUnit
from dsbox.utils import execute_dag


class TestUtils(unittest.TestCase):
    def test_execute_dag(self):
        # given
        dag = DAG(dag_id='test_exec_dag', start_date=datetime.now())

        tasks = []
        for i in range(0, 10):
            tasks.append(DummyOperator(task_id='task_' + str(i), dag=dag))

        tasks[0].set_downstream(tasks[1])
        tasks[1].set_downstream(tasks[2])
        tasks[2].set_downstream(tasks[3])
        tasks[2].set_downstream(tasks[4])
        tasks[3].set_downstream(tasks[5])
        tasks[4].set_downstream(tasks[5])
        tasks[4].set_downstream(tasks[8])
        tasks[8].set_downstream(tasks[9])

        tasks[6].set_downstream(tasks[7])

        # when
        task_list = execute_dag(dag)

        # then
        expected_task_lists = [
            [tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[8], tasks[9], tasks[6], tasks[7]],
            [tasks[6], tasks[7], tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[8], tasks[9]],
            [tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[8], tasks[9], tasks[5], tasks[6], tasks[7]],
            [tasks[6], tasks[7], tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[8], tasks[9], tasks[5]],
            ]

        expected_task_list_is_ok = (task_list in expected_task_lists)

        self.assertTrue(expected_task_list_is_ok)
