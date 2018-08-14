from abc import ABC, abstractmethod
import numpy as np


class PiBaseRegression(ABC):
    """
    this is a base class for all regressions
    in the Raspy library
    """
    @abstractmethod
    def __init__(self, X, y):
        """
        all supervised regressions will need an ``X`` and a ``y``.

        :param X: <np.ndarray> features
        :param y: <np.ndarray> labels
        """
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.y = y
        self.coefficients = None  # blank until they are populated by ``predict`` method.
        super(PiBaseRegression, self).__init__()

    @abstractmethod
    def fit(self):
        return self

    @abstractmethod
    def predict(self, X):
        return self


class PiLinearRegression(PiBaseRegression):
    """
    this is a linear regression algorithm for
    my raspberry pi model 3.
    """
    def __init__(self, X, y):
        super(PiLinearRegression, self).__init__(X=X, y=y)

    def fit(self):
        """
        fit linear regression
        """
        super().fit()
        self.coefficients = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.y))

    def predict(self, X):
        """
        given ``X`` this method predicts y_pred
        """
        super().predict(X=X)
        return X.dot(self.coefficients)
