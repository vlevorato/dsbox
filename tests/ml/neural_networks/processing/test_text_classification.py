import unittest

import numpy as np
from dsbox.ml.neural_networks.processing import Text2Sequence
from nltk.stem.snowball import EnglishStemmer

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)



np.random.seed(42)

class TestText2Sequence(unittest.TestCase):
    def test_TestText2Sequence_fit_and_transform_should_return_correct_sequences(self):
        # given
        X = np.array(['this is really awesome !',
                      'this is really crap !'])

        # when
        text2seq = Text2Sequence(stemmer=EnglishStemmer())
        sequences = text2seq.fit_transform(X)

        # then
        self.assertEqual(sequences[0][0], sequences[1][0])
        self.assertEqual(sequences[0][1], sequences[1][1])
        self.assertEqual(sequences[0][2], sequences[1][2])
        self.assertNotEqual(sequences[0][3], sequences[1][3])
        self.assertEqual(sequences[0][4], sequences[1][4])



if __name__ == '__main__':
    unittest.main()