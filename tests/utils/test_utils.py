import unittest
from datetime import datetime

from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator

from dsbox.utils import execute_dag, format_dict_path_items


class TestUtils(unittest.TestCase):
    def test_execute_dag(self):
        # given
        dag = DAG(dag_id='test_exec_dag', start_date=datetime.now())

        tasks = []
        for i in range(0, 11):
            tasks.append(DummyOperator(task_id='task_' + str(i), dag=dag))

        tasks[0].set_downstream(tasks[1])
        tasks[1].set_downstream(tasks[2])
        tasks[2].set_downstream(tasks[3])
        tasks[2].set_downstream(tasks[4])
        tasks[3].set_downstream(tasks[5])
        tasks[4].set_downstream(tasks[5])
        tasks[4].set_downstream(tasks[8])
        tasks[8].set_downstream(tasks[9])
        tasks[10].set_downstream(tasks[1])

        tasks[6].set_downstream(tasks[7])

        # when
        task_list = execute_dag(dag, mode='downstream')

        # then
        expected_task_lists = [
            [tasks[0], tasks[10], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[8], tasks[9], tasks[6],
             tasks[7]],
            [tasks[10], tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[8], tasks[9], tasks[6],
             tasks[7]],

            [tasks[6], tasks[7], tasks[0], tasks[10], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[8],
             tasks[9]],
            [tasks[6], tasks[7], tasks[10], tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[8],
             tasks[9]],

            [tasks[0], tasks[10], tasks[1], tasks[2], tasks[3], tasks[4], tasks[8], tasks[9], tasks[5], tasks[6],
             tasks[7]],
            [tasks[10], tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[8], tasks[9], tasks[5], tasks[6],
             tasks[7]],

            [tasks[6], tasks[7], tasks[0], tasks[10], tasks[1], tasks[2], tasks[3], tasks[4], tasks[8], tasks[9],
             tasks[5]],
            [tasks[6], tasks[7], tasks[10], tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[8], tasks[9],
             tasks[5]],

            [tasks[0], tasks[10], tasks[1], tasks[2], tasks[4], tasks[8], tasks[9], tasks[6], tasks[7], tasks[3],
             tasks[5]],
            [tasks[10], tasks[0], tasks[1], tasks[2], tasks[4], tasks[8], tasks[9], tasks[6], tasks[7], tasks[3],
             tasks[5]],

            [tasks[0], tasks[10], tasks[1], tasks[2], tasks[4], tasks[8], tasks[9], tasks[3], tasks[5], tasks[6],
             tasks[7]],
            [tasks[10], tasks[0], tasks[1], tasks[2], tasks[4], tasks[8], tasks[9], tasks[3], tasks[5], tasks[6],
             tasks[7]],

            [tasks[0], tasks[10], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6], tasks[7], tasks[8],
             tasks[9]],
            [tasks[10], tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5], tasks[6], tasks[7], tasks[8],
             tasks[9]],

            [tasks[0], tasks[10], tasks[1], tasks[2], tasks[4], tasks[3], tasks[5], tasks[6], tasks[7], tasks[8],
             tasks[9]],
            [tasks[10], tasks[0], tasks[1], tasks[2], tasks[4], tasks[3], tasks[5], tasks[6], tasks[7], tasks[8],
             tasks[9]],

            [tasks[10], tasks[6], tasks[7], tasks[0], tasks[1], tasks[2], tasks[4], tasks[3], tasks[8],  tasks[5],
             tasks[9]],
            [tasks[10], tasks[6], tasks[7], tasks[0], tasks[1], tasks[2], tasks[4], tasks[3], tasks[5], tasks[8],
             tasks[9]],
        ]

        print(task_list)

        expected_task_list_is_ok = (task_list in expected_task_lists)

        self.assertTrue(expected_task_list_is_ok)

    def test_format_dict_path_items(self):
        # given
        d = {'a': '{}/data',
             'b': {'c': '{}/data1', 'd': '{}/data2'},
             'l': [1, 3, 6],
             'k': ['{}/toto', 8.0],
             'o': False}

        r_value = 'mypath'

        # when
        d_result = format_dict_path_items(d, r_value)

        # then
        d_expected = {'a': 'mypath/data',
                      'b': {'c': 'mypath/data1', 'd': 'mypath/data2'},
                      'l': [1, 3, 6],
                      'k': ['mypath/toto', 8.0],
                      'o': False}

        self.assertDictEqual(d_expected, d_result)


if __name__ == '__main__':
    unittest.main()
