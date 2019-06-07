import logging
import unittest

import numpy as np
from keras.datasets import mnist

from dsbox.ml.neural_networks.keras_factory.autoencoders import DeepAutoEncoderFactory, AutoEncoderClassifier

logging.getLogger("tensorflow").setLevel(logging.WARNING)

np.random.seed(42)


class TestKerasTAutoEncoderModels(unittest.TestCase):
    def test_DeepAE_compilation_and_fit_predict_without_execution_error(self):
        # given
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = x_train[:100].astype('float32') / 255.
        x_test = x_test[:20].astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        # when
        ae_factory = DeepAutoEncoderFactory()
        autoencoder, encoder = ae_factory.create_model(start_layer_size=256, sub_layers_level=4)
        clf_keras = AutoEncoderClassifier(autoencoder, encoder)

        clf_keras.fit(x_train, x_train,
                      epochs=2,
                      batch_size=256,
                      shuffle=True,
                      validation_data=(x_test, x_test),
                      verbose=False)

        y_pred = clf_keras.predict(x_test)

        # then
        self.assertIsNotNone(y_pred)


if __name__ == '__main__':
    unittest.main()
