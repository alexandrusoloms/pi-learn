from abc import ABC, abstractmethod
import numpy as np


class PiBaseRegression(ABC):

    @abstractmethod
    def __init__(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.y = y
        self.coefficients = None
        super(PiBaseRegression, self).__init__()

    @abstractmethod
    def fit(self):
        return self

    @abstractmethod
    def predict(self, X):
        return X.dot(self.coefficients)
