from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator


class KubeDataOperator(KubernetesPodOperator):

    ui_color = '#defdff'

    def __init__(self, operation, kube_conf=None, **kwargs):
        self.operation = operation
        kube_conf['arguments'] += [operation]

        super(KubernetesPodOperator, self).__init__(**kwargs)
