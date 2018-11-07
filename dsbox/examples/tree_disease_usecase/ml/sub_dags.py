from airflow import DAG

from dsbox.examples.tree_disease_usecase.ml.feature_engineering import fillna_columns, category_to_numerical_features
from dsbox.operators.data_operator import DataOperator
from dsbox.operators.data_unit import DataInputFileUnit, DataOutputFileUnit
from dsbox.utils import FilenameGenerator


def feature_engineering_sub_dag(parent_dag_name, child_dag_name, temp_data_path, model_path, input_file, output_file,
                                start_date,
                                schedule_interval, mode='train'):
    filename_generator = FilenameGenerator(path=temp_data_path)
    temp_files = []
    for i in range(0, 100):
        temp_files.append(filename_generator.generate_filename() + '.parquet')

    dag = DAG('%s.%s' % (parent_dag_name, child_dag_name),
              schedule_interval=schedule_interval,
              start_date=start_date)

    task_fillna = DataOperator(operation_function=fillna_columns,
                               input_unit=DataInputFileUnit(input_file, pandas_read_function_name='read_parquet'),
                               output_unit=DataOutputFileUnit(temp_files[0], pandas_write_function_name='to_parquet'),
                               dag=dag, task_id='Fill_NA_values',
                               params={'model_path': model_path,
                                       'mode': mode}
                               )

    task_cat_to_num = DataOperator(operation_function=category_to_numerical_features,
                               input_unit=DataInputFileUnit(temp_files[0], pandas_read_function_name='read_parquet'),
                               output_unit=DataOutputFileUnit(output_file, pandas_write_function_name='to_parquet'),
                               dag=dag, task_id='Categorical_features_to_numeric',
                               params={'features' : ['GENRE_BOTA', 'ESPECE'],
                                        'model_path': model_path,
                                        'mode': mode}
                               )

    task_fillna.set_downstream(task_cat_to_num)

    return dag
