from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults


class DataOperator(BaseOperator):

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