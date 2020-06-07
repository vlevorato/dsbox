[![PyPI version](https://badge.fury.io/py/ds-box.svg)](https://badge.fury.io/py/ds-box)

# DS Box
<p align="center">
<img width="10%" src="https://user-images.githubusercontent.com/17388898/81501373-51baa880-92d8-11ea-8b96-d461bee1d21e.png">
</p>

Package made by <a href="https://www.linkedin.com/in/vlevorato/">V. Levorato</a>.

This package provides:
* specific **Operators for Apache Airflow** in order to manage Data Science workflows
* propose a **bunch of algorithms and routines** complementary to Data Science usual packages

## Install

DSbox can be installed from PyPI or by cloning this repo, and adding it to your project path.
```
pip install ds-box
```

## Airflow Operators and Data Units

Two specifics Airflow operators are given here: **DataOperator** and **KubeDataOperator**

### Airflow DataOperator
This operator is able to add input and/or output data during the operator execution. As Airflow operator are stateless, passing simple datasets to be read or written through, for instance, a PythonOperator, needs to write all the code inside the operator to achieve that. With a DataOperator, the only contract you have is to manipulate Pandas dataframe(s) inside the function passed to it. Here is the code structure:
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
  # useless function sorting dataframe by index
  return dataframe.sort_index()
 
reading_some_csv_unit = DataInputFileUnit('path/to/dataset.csv',
                                          pandas_read_function_name='read_csv',
                                          sep=';')
                                          
writing_some_parquet_unit = DataOutputFileUnit('path/to/index_sorted_dataset.parquet',
                                               pandas_write_function_name='to_parquet')                                     

```

Several pre-built ```DataUnit``` are avalaible and you can easily build yours. Some examples gives you a preview of complete dags workflow for data science.
`DataGlobalInputUnit` and `DataGlobalOutputUnit` can now be used with **any dataframe backend API (Pandas, Dask, Vaex, ...)** as you define explicitly which function you are using. 


### Airflow KubeDataOperator (KDO)

Using ```DataOperator``` can lead to some architecture issues: tasks must run on a machine or VM having the airflow worker service, and even if it should be moved to specific instance(s), it can be tricky to set auto-scaling mechanism, and to well manage your ressources.

To avoid that, you can use containers to execute your tasks in a Kubernetes cluster, with the same definition as ```DataOperator```, using ```KubeDataOperator```. This operator is based on the ```
KubernetesPodOperator```, and thought for data scientists.

The difference is that the operation definition is made in a YAML description file, and all you have to do is call the definition with this operator. It takes also all the Kubernetes configuration in one parameter (could be done with ```**kwargs``` but explicit is better). Same example as above with KDO:

```python
from dsbox.operators.kube_data_operator import KubeDataOperator

my_dag_task  = KubeDataOperator(operation='My_process',
                                kube_conf=build_kpo_config(),
                                dag=dag)
```

The ```build_kpo_config()``` function should be defined based on the project specificity and should explicitly set the ```cmds``` and ```arguments``` parameters needed by the underlying ```
KubernetesPodOperator```, in addition to the cluster configuration parameters. The ```cmds``` should call a function which will execute the task by using a class called ```Dataoperations``` containing all the datasets definitions with data units and operation functions. Example of a function called by ```cmds```:

```python
from dsbox.kube.data_operations import Dataoperations

def my_run_function(operation_name, data_root_path, data_operations_file_path):

    project_data_ops = Dataoperations(path=data_root_path)
    project_data_ops.load_datasets(data_operations_file_path)
    project_data_ops.run(operation_name)
```

and an example of the YAML data operations file that should be passed as ```data_operations_file_path```:
```yaml
My_process:
  operation_function:
    module: myprojectmodule.process
    name: my_function
  input_unit:
    type: DataInputFileUnit
    input_path: 'path/to/dataset.csv'
    pandas_read_function_name: read_csv
    sep: ';'
  output_unit:
    type: DataOutputFileUnit
    output_path: 'path/to/index_sorted_dataset.parquet'
    pandas_write_function_name: to_parquet
```

## Machine Learning part
 
This part is a humble contribution to accelerate Data Scientists recurrent operations, like for instance building lagged features for times series or exporting model features contribution. All the code produced here is Pandas and Scikit-learn friendly, as nearly everything is defined as a sklearn estimator.

### Tutorials 🎓
#### Neural Networks 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vlevorato/dsbox/blob/develop/notebooks/NeuralNets-AutoEncoder.ipynb) <a href="https://github.com/vlevorato/dsbox/blob/develop/notebooks/NeuralNets-AutoEncoder.ipynb">Classic Auto-encoders architectures</a>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vlevorato/dsbox/blob/develop/notebooks/NeuralNets-TextClassification.ipynb) <a href="https://github.com/vlevorato/dsbox/blob/develop/notebooks/NeuralNets-TextClassification.ipynb">
LSTM and CNN for text classification - Sentiment analysis applied to tweets in French</a>



### Subpackages list
 
 Subpackage | Description
------------ | -------------
Calibration | calibration model techniques
Ensemble | some ensemble models
Explain | model explanation methods
Feature_engineering | some categorical and time series pre-built feature engineering
Feature_selection | some feature selection methods
Markov | simple markov chains, supervised and unsupervised Hidden Markov Models
Neural_networks | end-to-end NN architectures
Outliers | some outlier detections techniques (MAD, FFT, Gaussian Process, etc.)
Text | some NLP models
Visualization | some pre-built function to display common graphics


_About copyrights on ML part_: some code has been re-packaged from existing libraries which are not or fewly maintained, and for which I could have been involved in the past. All licences are respected, all original authors and repos are quoted.

