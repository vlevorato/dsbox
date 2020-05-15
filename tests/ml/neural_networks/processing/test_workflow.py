import unittest

import numpy as np


import logging

from dsbox.ml.neural_networks.keras_factory.text_models import LSTMFactory
from dsbox.ml.neural_networks.processing.workflow import TextNeuralNetPipeline, ImageNeuralNetPipeline

logging.getLogger("tensorflow").setLevel(logging.WARNING)

np.random.seed(42)


class TestPipeline(unittest.TestCase):
    def test_fit_predict_text_nn_pipeline_should_return_some_result(self):
        # given
        x_train = np.array(['this is really really awesome !',
                            'it is so awesome !',
                            'that sucks']
                           )
        y_train = np.array([1, 1, 0])

        # when
        model = TextNeuralNetPipeline(factory_class=LSTMFactory, num_labels=2)
        model.fit(x_train, y_train, verbose=0)

        x_test = np.array(['it is really awesome !'])
        y_pred = model.predict(x_test)

        # then
        self.assertIsNotNone(y_pred)

    def test_fit_predict_proba_text_nn_pipeline_should_return_some_result(self):
        # given
        x_train = np.array(['this is really really awesome !',
                            'it is so awesome !',
                            'that sucks']
                           )
        y_train = np.array([1, 1, 0])

        # when
        model = TextNeuralNetPipeline(factory_class=LSTMFactory, num_labels=2)
        model.fit(x_train, y_train, verbose=0)

        x_test = np.array(['it is really awesome !'])
        y_pred = model.predict_proba(x_test)[0]

        # then
        self.assertIsNotNone(y_pred)

    def test_fit_image_nn_workflow_should_set_params_automatically(self):
        # given
        workflow = ImageNeuralNetPipeline(weights="imagenet")

        # when
        workflow.fit()

        # then
        self.assertTupleEqual((299, 299), workflow.img_size_)
        self.assertEqual("block14_sepconv2_act", workflow.last_conv_layer_name_)
        self.assertListEqual(["avg_pool", "predictions"], workflow.classifier_layer_names_)

