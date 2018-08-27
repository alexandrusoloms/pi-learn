from abc import ABC, abstractmethod
import numpy as np


class PiBaseRegression(ABC):

    @abstractmethod
    def __init__(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.y = y
        self.coefficients = None
        self.fit_intercept = None
        super(PiBaseRegression, self).__init__()

    @abstractmethod
    def fit(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        if not fit_intercept:
            self.X = self.X[:, 1:]
            self.n_features = self.n_features - 1
        return self

    @abstractmethod
    def predict(self, X):
        if not self.fit_intercept:
            return X[:, 1:].dot(self.coefficients)
        else:
            return X.dot(self.coefficients)

class PiBaseNonParametricRegression(ABC):

    @abstractmethod
    def __init__(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.y = y
        super(PiBaseNonParametricRegression, self).__init__()

    @abstractmethod
    def fit_predict(self, X):
        # not implemented here
        pass

class PiBaseClassifier(ABC):

    @abstractmethod
    def __init__(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.y = y
        self.coefficients = None
        self.fit_intercept = None
        super(PiBaseRegression, self).__init__()

    @abstractmethod
    def fit(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        if not fit_intercept:
            self.X = self.X[:, 1:]
            self.n_features = self.n_features - 1
        return self

    @abstractmethod
    def predict(self, X):
        if not self.fit_intercept:
            return X[:, 1:].dot(self.coefficients)
        else:
            return X.dot(self.coefficients)
