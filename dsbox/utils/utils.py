import gzip
import pickle
import networkx as nx
import matplotlib.pyplot as plt


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


def execute_dag(dag, verbose=False):
    task_list = []
    roots = dag.roots

    for root in roots:
        sub_task_list = breadth_first_search_task_list(root, task_list)
        task_list = sub_task_list + task_list

    task_list.reverse()

    for task in task_list:
        if verbose:
            print(str(task))
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
        color_list.append(node.ui_color)

    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw_networkx_nodes(G, pos, node_shape='D', node_color=color_list)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.8)

    nx.draw_networkx_labels(G, pos, font_size=5)

    ax.set_axis_off()
    plt.title("DAG preview", fontsize=8)
    plt.show()


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
