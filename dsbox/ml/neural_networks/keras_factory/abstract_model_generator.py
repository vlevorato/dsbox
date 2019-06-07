from abc import ABCMeta, abstractmethod


class KerasFactory:
    """
    Abstract class template for all keras neural nets factories
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_model(self, **kwargs):
        pass
