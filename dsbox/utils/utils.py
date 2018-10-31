import gzip
import pickle


def breadth_first_search_task_list(task_root):
    task_list = []
    queue = [task_root]

    while len(queue) > 0:
        task = queue.pop(0)
        task_list.append(task)
        next_tasks = task.downstream_list
        for next_task in next_tasks:
            if next_task not in queue and next_task not in task_list:
                queue.append(next_task)

    return task_list


def get_dag_roots(dag):
    roots = []
    for task in dag.tasks:
        if len(task.upstream_list) == 0:
            roots.append(task)

    return roots


def execute_dag(dag, verbose=False):
    task_list = []
    roots = get_dag_roots(dag)

    for root in roots:
        sub_task_list = breadth_first_search_task_list(root)
        task_list += sub_task_list

    for task in task_list:
        if verbose:
            print(str(task))
        task.execute(dag.get_template_env())

    return task_list


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
