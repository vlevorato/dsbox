from abc import ABCMeta, abstractmethod

__author__ = "Vincent Levorato"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"

class KerasFactory:
    """
    Abstract class template for all keras neural nets factories
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_model(self, **kwargs):
        pass
