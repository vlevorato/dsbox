import gzip
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from smart_open import open


def breadth_first_search_task_list(task_root, task_list=[], mode='upstream'):
    sub_task_list = []
    queue = [task_root]

    while len(queue) > 0:
        task = queue.pop(0)
        sub_task_list.append(task)
        next_tasks = None
        if mode == 'upstream':
            next_tasks = task.upstream_list
        else:
            next_tasks = task.downstream_list
        for next_task in next_tasks:
            if next_task not in queue and next_task not in task_list and next_task not in sub_task_list:
                queue.append(next_task)

    return sub_task_list


def breadth_first_search_shell_list(task_roots):
    shell_task_list = [task_roots]
    done_tasks = set()
    queue = task_roots

    while len(queue) > 0:
        tasks = queue
        next_tasks = []
        for task in tasks:
            for next_task in task.downstream_list:
                if next_task not in done_tasks:
                    next_tasks.append(next_task)
                    done_tasks.add(next_task)

        if len(next_tasks) > 0:
            shell_task_list.append(next_tasks)
        queue = next_tasks

    return shell_task_list


def get_dag_roots(dag):
    roots = []
    for task in dag.tasks:
        if len(task.upstream_list) == 0:
            roots.append(task)

    return roots


def execute_dag(dag, verbose=False, mode='downstream'):
    task_list = []
    roots = dag.roots

    for root in roots:
        sub_task_list = breadth_first_search_task_list(root, task_list, mode=mode)
        task_list = sub_task_list + task_list

    for task in task_list:
        if verbose:
            print(dag.dag_id + '-' + str(task))
        if task.task_type == 'SubDagOperator':
            execute_dag(task.subdag, verbose=verbose)
        else:
            task.execute(dag.get_template_env())

    return task_list


def plot_dag(dag):
    fig, ax = plt.subplots(figsize=(15, 10), dpi=150)

    G = nx.DiGraph()
    color_list = []

    for task in dag.tasks:
        if len(task.downstream_list) > 0:
            for next_task in task.downstream_list:
                G.add_edge(task, next_task)

    for node in G.nodes():
        if len(node.ui_color) == 7:
            color_list.append(node.ui_color)
        else:
            last_code = node.ui_color[-1]
            color_list.append(str(node.ui_color).ljust(7, last_code))

    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw_networkx_nodes(G, pos, node_shape='D', node_color=color_list)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.8)

    nx.draw_networkx_labels(G, pos, font_size=5)

    ax.set_axis_off()
    plt.title("DAG preview", fontsize=8)
    plt.show()


def pickle_compress(obj):
    return gzip.zlib.compress(pickle.dumps(obj))


def decompress_unpickle(obj_zp):
    return pickle.loads(gzip.zlib.decompress(obj_zp))


def write_object_file(file_path, obj):
    obj_pz = pickle_compress(obj)
    file_obj = open(file_path, 'wb')
    file_obj.write(obj_pz)
    file_obj.close()


def load_object_file(file_path):
    file_obj = open(file_path, 'rb')
    obj_pz = file_obj.read()
    obj = decompress_unpickle(obj_pz)
    return obj


def pandas_downcast_numeric(df_to_downcast, float_type_to_downcast=("float64", "float32"),
                            int_type_to_downcast=("int64", "int32")):
    float_cols = [c for c in df_to_downcast.columns if df_to_downcast[c].dtype == float_type_to_downcast[0]]
    int_cols = [c for c in df_to_downcast.columns if df_to_downcast[c].dtype == int_type_to_downcast[0]]
    df_to_downcast[float_cols] = df_to_downcast[float_cols].apply(lambda x: x.astype(float_type_to_downcast[1]))
    df_to_downcast[int_cols] = df_to_downcast[int_cols].apply(lambda x: x.astype(int_type_to_downcast[1]))


def format_dict_path_items(dictionary, replace_value):
    for k, v in dictionary.items():
        if isinstance(v, dict):
            dictionary[k] = format_dict_path_items(v, replace_value)
        else:
            if isinstance(v, list):
                formatted_list = []
                for list_item in v:
                    if type(list_item) == str:
                        list_item = list_item.format(replace_value)
                    formatted_list.append(list_item)
                dictionary[k] = formatted_list
            else:
                if type(dictionary[k]) == str:
                    dictionary[k] = dictionary[k].format(replace_value)
    return dictionary
