import pyarrow.plasma as plasma

test_user = 'vle'
test_password = 'tototo'
test_hostname = '192.168.253.161'
test_port = 5432
test_dbname = 'postgres'

socket_name="/tmp/plasma"

id = bytes([0x01] * 20)
object_id=plasma.ObjectID(id)