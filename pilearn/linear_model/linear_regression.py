import numpy as np
from ..base import PiBaseRegression


class PiLinearRegression(PiBaseRegression):
    """
    this is a linear regression algorithm for
    my raspberry pi model 3.
    """
    def __init__(self, X, y):
        super(PiLinearRegression, self).__init__(X=X, y=y)

    def fit(self, fit_intercept=True):
        """
        fit linear regression
        """
        super().fit(fit_intercept=fit_intercept)
        self.coefficients = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.y))

    def predict(self, X):
        """
        given ``X`` this method predicts y_pred
        """
        return super().predict(X=X)
