class DataExecutor:

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
