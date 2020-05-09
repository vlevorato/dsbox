import warnings

import numpy as np
import pyarrow as pa
import pyarrow.plasma as plasma

__status__ = "Beta"

class PlasmaConnector:
    """
    Based on PyArrow documentation
    """
    def __init__(self, socket_name, manager_socket_name="", int_release_delay=0):
        warnings.warn("beta version", Warning)
        self.socket_name = socket_name
        self.manager_socket_name = manager_socket_name
        self.int_release_delay = int_release_delay

        self.client = plasma.connect(socket_name, manager_socket_name, int_release_delay)

    @staticmethod
    def generate_object_id():
        return plasma.ObjectID(np.random.bytes(20))

    def put_dataframe(self, dataframe, object_id=None, overwrite=True):
        if object_id is None:
            object_id = PlasmaConnector.generate_object_id()

        # Delete object if exists
        if overwrite:
            self.client.delete([object_id])
        else:
            if self.client.contains(object_id):
                raise ValueError("object id already exists.")

        # Convert the Pandas DataFrame into a PyArrow RecordBatch
        record_batch = pa.RecordBatch.from_pandas(dataframe)

        # Create the Plasma object from the PyArrow RecordBatch. Most of the work here
        # is done to determine the size of buffer to request from the object store.
        mock_sink = pa.MockOutputStream()
        stream_writer = pa.RecordBatchStreamWriter(mock_sink, record_batch.schema)
        stream_writer.write_batch(record_batch)
        stream_writer.close()

        data_size = mock_sink.size()
        buffer = self.client.create(object_id, data_size)

        # Write the PyArrow RecordBatch to Plasma
        stream = pa.FixedSizeBufferWriter(buffer)
        stream_writer = pa.RecordBatchStreamWriter(stream, record_batch.schema)
        stream_writer.write_batch(record_batch)
        stream_writer.close()

        # Seal the Plasma object (make it immutable and usable by others clients)
        self.client.seal(object_id)

        return object_id

    def get_dataframe(self, object_id):
        # Fetch the Plasma object
        [data] = self.client.get_buffers([object_id])  # Get PlasmaBuffer from ObjectID
        buffer = pa.BufferReader(data)

        # Convert object back into an Arrow RecordBatch
        reader = pa.RecordBatchStreamReader(buffer)
        record_batch = reader.read_next_batch()

        # Convert back into Pandas
        return record_batch.to_pandas()
