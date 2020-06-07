import importlib

from dsbox.ml.neural_networks import KerasFactory

__author__ = "Vincent Levorato"
__license__ = "Apache 2.0"


class KerasApplicationFactory(KerasFactory):
    def create_model(self, module_name='tensorflow.keras.applications.xception', model_name='Xception', **kwargs):
        """
        Build a keras pre-trained model


        Parameters
        ----------
        module_name: str, optional (default='tensorflow.keras.applications')
            Keras module containing pre-built models
        model_name: str, optional (default='Xception')
            Keras pre-built model
        **kwargs :
            parameters passed to inner keras model

        Returns
        -------
        Keras model

        .. Keras pre-trained models list: https://keras.io/api/applications/

        """
        module = importlib.import_module(module_name)
        model_builder = getattr(module, model_name)

        return model_builder(**kwargs)
