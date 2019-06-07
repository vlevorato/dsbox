import logging
import unittest

import numpy as np
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

logging.getLogger("tensorflow").setLevel(logging.WARNING)

from dsbox.ml.neural_networks.keras_factory.text_models import LSTMFactory

np.random.seed(42)


def dummy_hash_function(x):
    code = 0
    for c in x:
        code += ord(c)
    return code


class TestKerasTextModels(unittest.TestCase):
    def test_LSTM_compilation_and_fit_predict_without_execution_error(self):
        # given
        x_train = np.array(['this is really awesome !',
                            'this is really crap !!']
                           )
        y_train = np.array([1, 0])

        ids_x_train = np.empty([2, 5])
        for i in range(0, len(x_train)):
            ids = [dummy_hash_function(token) for token in x_train[i].split(' ')]
            ids_x_train[i, :] = ids
        num_labels = 2
        y_enc = np_utils.to_categorical(y_train, num_labels)
        dictionary_size = np.int(np.max(ids_x_train) + 1)

        # when
        lstm_factory = LSTMFactory()
        clf_keras = KerasClassifier(build_fn=lstm_factory.create_model,
                                    dictionary_size=dictionary_size,
                                    num_labels=num_labels)
        clf_keras.fit(ids_x_train, y_enc, epochs=1, verbose=False)

        x_test = np.array(['it is really awesome !'])
        ids_x_test = np.empty([1, 5])
        ids_x_test[0, :] = [dummy_hash_function(token) for token in x_test[0].split(' ')]

        y_pred = clf_keras.predict(ids_x_test)

        # then
        self.assertIsNotNone(y_pred)


if __name__ == '__main__':
    unittest.main()
