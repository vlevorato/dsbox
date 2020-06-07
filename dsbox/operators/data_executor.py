class DataExecutor:
    """
    Same behavior as the Data Operator, without Airflow dependency. Based on Data Units, it can be used to execute
    operation directly by calling the execute() method.

    Parameters
    ----------
    operation_function: function
        function taking the output of the input_unit and returning its result to the input of the output_unit
    input_unit: DataInputUnit, default=None
        define input data reading
    output_unit: DataOutputUnit, default=None
        define output data writing
    """

    def __init__(self, operation_function, input_unit=None, output_unit=None, **kwargs):
        self.operation_function = operation_function
        self.input_unit = input_unit
        self.output_unit = output_unit
        self.kwargs = kwargs

    def execute(self):
        data = None
        if self.input_unit is not None:
            data = self.input_unit.read_data()
            data = self.operation_function(data, **self.kwargs)
        else:
            data = self.operation_function(**self.kwargs)
        if self.output_unit is not None:
            self.output_unit.write_data(data)
