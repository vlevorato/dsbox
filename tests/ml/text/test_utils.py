import unittest

from dsbox.ml.text.utils import shinglelize


class TestTextFeatureExtraction(unittest.TestCase):
    def test_shinglelize_function(self):
        # given
        text = "There is a better way."

        # when
        shingles_obtained = shinglelize(text)

        # then
        expected_shingle_list = ['The', 'her', 'ere', 're ', 'e i', ' is', 'is ', 's a', ' a ', 'a b', ' be', 'bet',
                                 'ett', 'tte', 'ter', 'er ', 'r w', ' wa', 'way', 'ay.']
        self.assertEqual(expected_shingle_list, shingles_obtained)

    def test_shinglelize_function_on_empty_string(self):
        # given
        text = ""

        # when
        shingles_obtained = shinglelize(text)

        # then
        self.assertEqual(shingles_obtained, [])


if __name__ == '__main__':
    unittest.main()
