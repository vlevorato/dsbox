from datetime import datetime
import os

from airflow import DAG

from dsbox.examples.sales_prediction_usecase.ml.feature_engineering import concat_train_test, resample_fillna, \
    create_simple_features, create_timeseries_rolling_features, create_timeseries_shift_features, \
    create_timeseries_diff_shift_features, merge_time_features, extra_features, merge_final_features
from dsbox.examples.sales_prediction_usecase.ml.modeling import train_model, metrics_model, predict_model
from dsbox.operators.data_operator import DataOperator
from dsbox.operators.data_unit import DataInputFileUnit, DataOutputFileUnit, DataInputMultiFileUnit

"""
This usecase is taken from the Kaggle Rossmann challenge.

"""

project_path = os.getenv('PROJECT_PATH')

dag = DAG(dag_id='Sales_Prediction', description='Sales Prediction Example',
          schedule_interval='0 20 * * *', start_date=datetime(2017, 3, 20), catchup=False)

"""
Concat train and test data (easier for feature engineering when facing a timeseries problem.
"""
input_parquet_files_unit = DataInputMultiFileUnit([project_path + 'datasets/input/train.parquet',
                                                   project_path + 'datasets/input/test.parquet'],
                                                  pandas_read_function_name='read_parquet')

output_parquet_concat_unit = DataOutputFileUnit(project_path + 'datasets/temp/X_raw.parquet',
                                                pandas_write_function_name='to_parquet')

task_concat_train_files = DataOperator(operation_function=concat_train_test,
                                       input_unit=input_parquet_files_unit,
                                       output_unit=output_parquet_concat_unit,
                                       dag=dag, task_id='Concat_train_test_data_source_files')

"""
Resampling time data
"""
input_raw_data_unit = DataInputFileUnit(output_parquet_concat_unit.output_path,
                                        pandas_read_function_name='read_parquet')

output_cleaned_data_unit = DataOutputFileUnit(project_path + 'datasets/temp/X_clean.parquet',
                                              pandas_write_function_name='to_parquet')

task_fill_missing_values = DataOperator(operation_function=resample_fillna,
                                        input_unit=input_raw_data_unit,
                                        output_unit=output_cleaned_data_unit,
                                        dag=dag, task_id='Resample_and_fill_NA_values')

task_concat_train_files.set_downstream(task_fill_missing_values)

"""
Simple feature engineering 
"""

input_cleaned_data_unit = DataInputFileUnit(output_cleaned_data_unit.output_path,
                                            pandas_read_function_name='read_parquet')

output_simple_fe_data_unit = DataOutputFileUnit(project_path + 'datasets/temp/X_simple_fe.parquet',
                                                pandas_write_function_name='to_parquet')

task_simple_fe = DataOperator(operation_function=create_simple_features,
                              input_unit=input_cleaned_data_unit,
                              output_unit=output_simple_fe_data_unit,
                              dag=dag, task_id='Simple_feature_engineering')

task_fill_missing_values.set_downstream(task_simple_fe)

"""
Time feature engineering
"""

operations = ['mean', 'std', 'median', 'min', 'max']
shift_ranges = [42, 49, 56, 63] + list(range(70, 364, 28)) + [364]
rolling_windows = [7, 28]

input_cleaned_sub_data_unit = DataInputFileUnit(output_simple_fe_data_unit.output_path,
                                                pandas_read_function_name='read_parquet',
                                                columns=['Date', 'Store', 'Sales',
                                                         'Customers', 'SalePerCustomer'])
rolling_tasks = []

for operation in operations:
    output_rolling_data_unit = DataOutputFileUnit(project_path + 'datasets/temp/X_' + operation + '.parquet',
                                                  pandas_write_function_name='to_parquet')

    task_rolling = DataOperator(operation_function=create_timeseries_rolling_features,
                                params={'features': ['Sales', 'Customers', 'SalePerCustomer'],
                                        'operation': operation,
                                        'shift_ranges': shift_ranges,
                                        'rolling_windows': rolling_windows},
                                input_unit=input_cleaned_sub_data_unit,
                                output_unit=output_rolling_data_unit,
                                dag=dag, task_id='Rolling_' + operation + '_feature_engineering')

    rolling_tasks.append(task_rolling)
    task_simple_fe.set_downstream(task_rolling)

output_shift_data_unit = DataOutputFileUnit(project_path + 'datasets/temp/X_shift.parquet',
                                            pandas_write_function_name='to_parquet')

task_shift = DataOperator(operation_function=create_timeseries_shift_features,
                          params={'features': ['Sales', 'Customers', 'SalePerCustomer'],
                                  'shift_ranges': shift_ranges},
                          input_unit=input_cleaned_sub_data_unit,
                          output_unit=output_shift_data_unit,
                          dag=dag, task_id='Time_shift_feature_engineering')

task_simple_fe.set_downstream(task_shift)

input_shift_data_unit = DataInputFileUnit(output_shift_data_unit.output_path,
                                          pandas_read_function_name='read_parquet')
output_diff_shift_data_unit = DataOutputFileUnit(project_path + 'datasets/temp/X_diff_shift.parquet',
                                                 pandas_write_function_name='to_parquet')

task_diff_shift = DataOperator(operation_function=create_timeseries_diff_shift_features,
                               params={'features': ['Sales', 'Customers', 'SalePerCustomer'],
                                       'shift_ranges': shift_ranges},
                               input_unit=input_shift_data_unit,
                               output_unit=output_diff_shift_data_unit,
                               dag=dag, task_id='Time_diff_shift_feature_engineering')

task_shift.set_downstream(task_diff_shift)

paths_to_merge = [output_simple_fe_data_unit.output_path,
                  output_shift_data_unit.output_path,
                  output_diff_shift_data_unit.output_path]
for operation in operations:
    paths_to_merge.append(project_path + 'datasets/temp/X_' + operation + '.parquet')

input_time_files_unit = DataInputMultiFileUnit(paths_to_merge, pandas_read_function_name='read_parquet')

output_time_data_unit = DataOutputFileUnit(project_path + 'datasets/temp/X_time.parquet',
                                           pandas_write_function_name='to_parquet')

task_merge_time_features = DataOperator(operation_function=merge_time_features,
                                        params={'features': ['Sales', 'Customers', 'SalePerCustomer']},
                                        input_unit=input_time_files_unit,
                                        output_unit=output_time_data_unit,
                                        dag=dag, task_id='Time_merge_feature_engineering')

task_diff_shift.set_downstream(task_merge_time_features)
for rolling_task in rolling_tasks:
    rolling_task.set_downstream(task_merge_time_features)

"""
Extra data feature engineering
"""
input_extra_data_unit = DataInputFileUnit(project_path + 'datasets/input/store.csv')
output_extra_data_unit = DataOutputFileUnit(project_path + 'datasets/temp/X_extra.parquet',
                                            pandas_write_function_name='to_parquet')

task_extra_features = DataOperator(operation_function=extra_features,
                                   input_unit=input_extra_data_unit,
                                   output_unit=output_extra_data_unit,
                                   dag=dag, task_id='Extra_data_feature_engineering')

"""
Final features merge
"""
input_merge_files_unit = DataInputMultiFileUnit([project_path + 'datasets/temp/X_time.parquet',
                                                 project_path + 'datasets/temp/X_extra.parquet'],
                                                pandas_read_function_name='read_parquet')
output_merge_data_unit = DataOutputFileUnit(project_path + 'datasets/temp/X_final.parquet',
                                            pandas_write_function_name='to_parquet')

task_final_features = DataOperator(operation_function=merge_final_features,
                                   input_unit=input_merge_files_unit,
                                   output_unit=output_merge_data_unit,
                                   dag=dag, task_id='Final_data_merge')

task_extra_features.set_downstream(task_final_features)
task_merge_time_features.set_downstream(task_final_features)

"""
Train model
"""
input_final_data_unit = DataInputFileUnit(output_merge_data_unit.output_path,
                                          pandas_read_function_name='read_parquet')

features_to_exclude = ['Datetime', 'Date', 'Sales', 'Customers', 'SalePerCustomer', 'Id', 'to_predict', 'Open',
                       'MissingValues']

task_train_model = DataOperator(operation_function=train_model,
                                params={'features_to_exclude': features_to_exclude,
                                        'model_path': project_path + 'models/'},
                                input_unit=input_final_data_unit,
                                dag=dag, task_id='Train_model')

task_final_features.set_downstream(task_train_model)

"""
Metrics and prediction
"""

task_metrics = DataOperator(operation_function=metrics_model,
                            params={'features_to_exclude': features_to_exclude},
                            input_unit=input_final_data_unit,
                            dag=dag, task_id='Metrics')

task_train_model.set_downstream(task_metrics)

output_predict_data_unit = DataOutputFileUnit(project_path + 'datasets/output/X_predict.csv', index=False)

task_predict = DataOperator(operation_function=predict_model,
                            params={'features_to_exclude': features_to_exclude,
                                    'model_path': project_path + 'models/'},
                            input_unit=input_final_data_unit,
                            output_unit=output_predict_data_unit,
                            dag=dag, task_id='Prediction')

task_train_model.set_downstream(task_predict)
