#!/usr/bin/env bash
source activate airflow
cp $PROJECT_PATH/tests/*dag*.py $AIRFLOW_HOME/dags
