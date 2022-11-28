from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartPodOperator


class KubeDataOperator(KubernetesPodOperator):
    """

    Parameters
    ----------
    operation: str
        operation name defined in the data catalog operations used in the project

    kube_conf: dict, default=None
        Kubernetes cluster configuration. Must contain at least 'arguments' and 'cmds' keys

    kwargs: dict
        Other Parameters passed to KubernetesPodOperator constructor
    """

    ui_color = '#defdff'

    def __init__(self, operation, kube_conf=None, **kwargs):
        self.operation = operation
        kube_conf['arguments'] += [operation]
        kube_conf['task_id'] = operation
        kube_conf.update(kwargs)

        name = '{}-{}-pod'.format(kube_conf['dag'].dag_id, operation).replace('_', '-').lower()[0:63]
        kube_conf['name'] = name

        super().__init__(**kube_conf)


class GKEDataOperator(GKEStartPodOperator):
    """

    Parameters
    ----------
    operation: str
        operation name defined in the data catalog operations used in the project

    kube_conf: dict, default=None
        Kubernetes cluster configuration. Must contain at least 'arguments' and 'cmds' keys

    kwargs: dict
        Other Parameters passed to GKEStartPodOperator constructor
    """

    ui_color = '#defdff'

    def __init__(self, operation, kube_conf=None, **kwargs):
        self.operation = operation
        kube_conf['arguments'] += [operation]
        kube_conf['task_id'] = operation
        kube_conf.update(kwargs)

        name = '{}-{}-pod'.format(kube_conf['dag'].dag_id, operation).replace('_', '-').lower()[0:63]

        kube_conf['name'] = name

        super().__init__(**kube_conf)
