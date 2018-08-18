from abc import ABC, abstractmethod


class PiBasePreprocessing(ABC):
    """
    Base Class for preprocessing sub-directory.
    """
    @abstractmethod
    def __init__(self, X):
        super(PiBasePreprocessing, self).__init__()
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.list_of_arrays = list()
        self.transformed = None  # method not implemented in ABC

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass
