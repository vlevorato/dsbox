import gzip
import pickle


def get_downstream_list(task, task_list, level=0):
    task_list.append(task)
    level += 1
    for t in task.upstream_list:
        get_downstream_list(t, task_list, level)


def execute_dag(dag, verbose=False):
    task_list = []
    for t in dag.roots:
        get_downstream_list(t, task_list)

    task_list.reverse()
    for task in task_list:
        if verbose:
            print(str(task))
        task.execute(dag.get_template_env())


def pickle_compress(model):
    return gzip.zlib.compress(pickle.dumps(model))


def decompress_unpickle(model_zp):
    return pickle.loads(gzip.zlib.decompress(model_zp))


def write_model_file(file_path, model):
    model_pz = pickle_compress(model)
    file_obj = open(file_path, 'wb')
    file_obj.write(model_pz)
    file_obj.close()


def load_model_file(file_path):
    file_obj = open(file_path, 'rb')
    model_pz = file_obj.read()
    model = decompress_unpickle(model_pz)
    return model
