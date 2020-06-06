from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults


class DataOperator(BaseOperator):
    """

    Data Operator is a custom operator which can handle input and/or output data.
    The parameters of the operation function has to be passed as a dict named 'params', based on BaseOperator behavior.

    Parameters
    ----------
    operation_function: function
        function taking the output of the input_unit and returning its result to the input of the output_unit
    input_unit: DataInputUnit, default=None
        define input data reading
    output_unit: DataOutputUnit, default=None
        define output data writing
    """

    ui_color = '#fff2e6'

    @apply_defaults
    def __init__(self, operation_function, input_unit=None, output_unit=None, *args, **kwargs):
        self.operation_function = operation_function
        self.input_unit = input_unit
        self.output_unit = output_unit

        super(DataOperator, self).__init__(*args, **kwargs)

    def execute(self, context):
        data = None
        if self.input_unit is not None:
            data = self.input_unit.read_data()
            data = self.operation_function(data, **self.params)
        else:
            data = self.operation_function(**self.params)
        if self.output_unit is not None:
            self.output_unit.write_data(data)
