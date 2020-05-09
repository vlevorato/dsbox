from collections import OrderedDict

__author__ = "Vincent Levorato"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


class HistogramData:
    """
    Class which represents the frequency distribution of given data.
    
    """
    def __init__(self, intervals, data):
        self.interval_dict = OrderedDict()  # structure: key (if value > key and value < key+1) => amount
        self.amount_data = 0

        for index, element in enumerate(data):
            self.interval_dict[intervals[index]] = element
            self.amount_data += element

        self.interval_dict[intervals[len(intervals) - 1]] = 0

    def get_amount(self, value, normalized=False, cumulative=None):
        keys = list(self.interval_dict.keys())

        sum_values = 0

        if len(keys) == 0:
            return 0
        if len(keys) == 1:
            amount = 0
            if self.interval_dict.get(keys[0]) is not None:
                amount = self.interval_dict[keys[0]]
                if normalized:
                    if self.amount_data != 0:
                        amount = float(self.interval_dict[keys[0]]) / self.amount_data

            return amount

        found = False
        i = len(keys) - 1

        while not found:
            if value >= keys[i]:
                found = True
            else:
                if self.interval_dict.get(keys[i]) is not None:
                    sum_values += self.interval_dict[keys[i]]
                i -= 1
                if i < 0:
                    return 0

        divisor = 1.0
        if normalized:
            divisor = self.amount_data

        if self.interval_dict.get(keys[i]) is not None:
            if cumulative is None:
                return float(self.interval_dict[keys[i]]) / divisor
            if cumulative == 'desc':
                return float(sum_values) / divisor
            if cumulative == 'asc':
                return float(self.amount_data - sum_values) / divisor
        else:
            return 0
