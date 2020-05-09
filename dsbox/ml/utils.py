__author__ = "Vincent Levorato"
__license__ = "Apache 2.0"


def check_estimator_predict_proba(estimator):
    if not hasattr(estimator, "predict_proba"):
        raise RuntimeError('classifier has no predict_proba method.')