import os
import pyarrow.plasma as plasma

test_user = os.environ.get(['TEST_DSBOX_PG_USER'])
test_password = os.environ.get(['TEST_DSBOX_PG_PASSWD'])
test_hostname = os.environ.get(['TEST_DSBOX_PG_HOST'])
test_port = os.environ.get(['TEST_DSBOX_PG_PORT'])
test_dbname = os.environ.get(['TEST_DSBOX_PG_DBNAME'])

socket_name = os.environ.get(['TEST_DSBOX_PLASMA_SOCKET_NAME'])

id = bytes([0x01] * 20)
object_id = plasma.ObjectID(id)