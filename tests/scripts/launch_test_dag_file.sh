#!/usr/bin/env bash
. copy_dag_files.sh
source activate airflow
PYTHONPATH="${PYTHONPATH}:$PROJECT_PATH"
export PYTHONPATH
cd $PROJECT_PATH/tests/
airflow backfill -s -1 test_dag_file
