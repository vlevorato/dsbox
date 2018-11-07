from datetime import datetime
import os

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.subdag_operator import SubDagOperator

from dsbox.examples.tree_disease_usecase.ml.feature_engineering import join_dataframes, fillna_columns
from dsbox.examples.tree_disease_usecase.ml.modeling import fit_write_model, read_predict_model, model_performance
from dsbox.examples.tree_disease_usecase.ml.sub_dags import feature_engineering_sub_dag
from dsbox.operators.data_operator import DataOperator
from dsbox.operators.data_unit import DataInputFileUnit, DataOutputFileUnit, DataInputMultiFileUnit
from dsbox.utils import execute_dag, plot_dag
from dsbox.utils import FilenameGenerator

from sqlalchemy import create_engine

engine = create_engine('sqlite:///tree_disease.db', echo=False)

data_path = os.getenv('DATA_PATH')


def dummy_function(dataframe):
    return dataframe


features_selection = ['ADR_SECTEUR', 'ANNEEDEPLANTATION', 'coord_x', 'coord_y',
                      'ANNEEREALISATIONDIAGNOSTIC', 'ANNEETRAVAUXPRECONISESDIAG',
                      'GENRE_BOTA', 'ESPECE', 'DIAMETREARBREAUNMETRE']

feature_target = 'Default'
prediction_column_name = 'y_prediction'

filename_generator = FilenameGenerator(path='datasets/temp/')
temp_files = []
for i in range(0, 100):
    temp_files.append(filename_generator.generate_filename() + '.parquet')

dag = DAG(dag_id='Tree_Disease_Prediction', start_date=datetime.now())

input_csv_files_unit = DataInputMultiFileUnit(['datasets/input/X_tree_egc_t1.csv',
                                               'datasets/input/X_geoloc_egc_t1.csv',
                                               'datasets/input/Y_tree_egc_t1.csv'], sep=';')
output_parquet_unit = DataOutputFileUnit('datasets/temp/X_train_raw.parquet', pandas_write_function_name='to_parquet')
task_concate_train_files = DataOperator(operation_function=join_dataframes,
                                        input_unit=input_csv_files_unit,
                                        output_unit=output_parquet_unit,
                                        dag=dag, task_id='Join_train_data_source_files')

task_feature_engineering_for_train = SubDagOperator(
    subdag=feature_engineering_sub_dag(dag.dag_id, 'Feature_engineering_for_train',
                                       temp_data_path='datasets/temp/',
                                       model_path='models/',
                                       input_file='datasets/temp/X_train_raw.parquet',
                                       output_file='datasets/temp/X_train_final.parquet',
                                       start_date=dag.start_date,
                                       schedule_interval=dag.schedule_interval),
    task_id='Feature_engineering_for_train',
    dag=dag,
)

task_concate_train_files.set_downstream(task_feature_engineering_for_train)

input_parquet_raw_file_unit = DataInputFileUnit('datasets/temp/X_train_final.parquet',
                                                pandas_read_function_name='read_parquet')
task_model_learning = DataOperator(operation_function=fit_write_model,
                                   params={'columns_selection': features_selection,
                                           'column_target': feature_target,
                                           'write_path': 'models/tree.model'
                                           },
                                   input_unit=input_parquet_raw_file_unit,
                                   dag=dag, task_id='Model_learning')

task_feature_engineering_for_train.set_downstream(task_model_learning)

input_csv_files_unit = DataInputMultiFileUnit(['datasets/input/X_tree_egc_t2.csv',
                                               'datasets/input/X_geoloc_egc_t2.csv',
                                               'datasets/input/Y_tree_egc_t2.csv'], sep=';')
output_parquet_unit = DataOutputFileUnit('datasets/temp/X_test_raw.parquet', pandas_write_function_name='to_parquet')
task_concate_test_files = DataOperator(operation_function=join_dataframes,
                                       input_unit=input_csv_files_unit,
                                       output_unit=output_parquet_unit,
                                       dag=dag, task_id='Join_test_data_source_files')

task_feature_engineering_for_test = SubDagOperator(
    subdag=feature_engineering_sub_dag(dag.dag_id, 'Feature_engineering_for_test',
                                       temp_data_path='datasets/temp/',
                                       model_path='models/',
                                       input_file='datasets/temp/X_test_raw.parquet',
                                       output_file='datasets/temp/X_test_final.parquet',
                                       start_date=dag.start_date,
                                       schedule_interval=dag.schedule_interval,
                                       mode='predict'),
    task_id='Feature_engineering_for_test',
    dag=dag,
)

task_concate_test_files.set_downstream(task_feature_engineering_for_test)

task_model_predict = DataOperator(operation_function=read_predict_model,
                                  params={'columns_selection': features_selection,
                                          'read_path': 'models/tree.model',
                                          'y_pred_column_name': prediction_column_name
                                          },
                                  input_unit=DataInputFileUnit('datasets/temp/X_test_final.parquet',
                                                               pandas_read_function_name='read_parquet'),
                                  output_unit=DataOutputFileUnit('datasets/output/X_predict.csv',
                                                                 pandas_write_function_name='to_csv',
                                                                 index=False),
                                  dag=dag, task_id='Model_prediction')

task_feature_engineering_for_test.set_downstream(task_model_predict)
task_model_learning.set_downstream(task_model_predict)

input_result_file_unit = DataInputFileUnit('datasets/output/X_predict.csv', pandas_read_function_name='read_csv')
task_model_metric = DataOperator(operation_function=model_performance,
                                 params={'y_true_column_name': feature_target,
                                         'y_pred_column_name': prediction_column_name
                                         },
                                 input_unit=input_result_file_unit,
                                 dag=dag, task_id='Model_metrics')

task_model_predict.set_downstream(task_model_metric)

output_result_unit = DataOutputFileUnit('predictions', pandas_write_function_name='to_sql',
                                        con=engine,
                                        if_exists='replace')
task_export_to_sqlite = DataOperator(operation_function=dummy_function,
                                     input_unit=input_result_file_unit,
                                     output_unit=output_result_unit,
                                     dag=dag, task_id='Export_result_to_sqlite')

task_model_predict.set_downstream(task_export_to_sqlite)

task_purge_temp_files = BashOperator(task_id='Purge_temp_files',
                                     bash_command='rm ' + data_path + 'temp/*',
                                     dag=dag)

task_export_to_sqlite.set_downstream(task_purge_temp_files)

# dag.tree_view()
#plot_dag(dag)
print()

execute_dag(dag, verbose=True)
