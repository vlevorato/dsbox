import unittest

from dsbox.ml.markov.histrogramdata import HistogramData


class TestHistogramData(unittest.TestCase):
    def test_get_amount_method_should_return_correct_values(self):
        # given
        data = [55, 28, 37]
        intervals = [0, 5, 10, 15]  # histogram intervals

        # when
        hda = HistogramData(intervals, data)

        # then
        amount = hda.get_amount(6)
        self.assertEqual(amount, 28)

        amount = hda.get_amount(11)
        self.assertEqual(amount, 37)

        amount = hda.get_amount(1)
        self.assertEqual(amount, 55)

        amount = hda.get_amount(-1)
        self.assertEqual(amount, 0)

    def test_get_amount_normalized_should_return_correct_values(self):
        # given
        data = [55, 28, 37]
        intervals = [0, 5, 10, 15]  # histogram intervals

        # when
        hda = HistogramData(intervals, data)
        amount = hda.get_amount(6, normalized=True)

        # then
        self.assertEqual(amount, float(28) / sum(data))
