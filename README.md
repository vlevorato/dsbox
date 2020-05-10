# DS Box
<p align="center">
<img width="10%" src="https://user-images.githubusercontent.com/17388898/81501373-51baa880-92d8-11ea-8b96-d461bee1d21e.png">
</p>

Package made by <a href="https://www.linkedin.com/in/vlevorato/">V. Levorato</a> .

This package provides:
* specific **Operators for Apache Airflow** in order to manage Data Science workflows
* propose a **bunch of algorithms and routines** complementary to Data Science usual packages


## Airflow Operators and Data Units

Two specifics Airflow operators are given here: **DataOperator** and **KubeDataOperator**

### Airfow DataOperator
This operator is able to add input and/or output data during the operator execution. As Airflow operator are stateless, passing simpe datasets to be read or written through, for instance, a PythonOperator, needs to write everything. With a DataOperator, the only contract you have is to manipulate Pandas dataframe(s) inside the function passed to it. Here the code structure:
```python
from dsbox.operators.data_operator import DataOperator

my_dag_task = DataOperator(operation_function=my_function,
                           input_unit=reading_some_csv_unit,
                           output_unit=writing_some_parquet_unit,
                           dag=dag, task_id='My_process')
```

With this example, I just need to define ```my_function```, the ```input_unit``` and the ```output_unit```.

```python
from dsbox.operators.data_unit import DataInputFileUnit, DataOutputFileUnit

def my_function(dataframe):
  return dataframe.sort_index()
 
reading_some_csv_unit = DataInputFileUnit('path/to/dataset.csv',
                                          pandas_read_function_name='read_csv',
                                          sep=';')
                                          
writing_some_parquet_unit = DataOutputFileUnit('path/to/index_sorted_dataset.parquet',
                                               pandas_write_function_name='to_parquet')                                     








_About copyrights on ML part_: some code has been re-packaged from existing libraries which are not or fewly maintained, and for which I could have been involved in the past. All licences are respected and original authors and repos quoted.


 

