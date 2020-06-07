import sys

from dsbox.kube.data_operations import Dataoperations

"""
Main program that should be used by a KubeDataOperator by passing it as 'arguments' 
parameters to execute operation on the data during the Pod execution.
"""

if __name__ == "__main__":
    assert len(sys.argv) > 1

    operation_name = sys.argv[1]
    path = sys.argv[2]
    operations_path = sys.argv[3]

    project_data_ops = Dataoperations(path=path)

    project_data_ops.load_datasets(operations_path)

    project_data_ops.run(operation_name)
