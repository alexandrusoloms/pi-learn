import numpy as np
from ..base import PiBaseRegression


class PiRidgeRegression(PiBaseRegression):
    """
    Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of
    coefficients.
    The ridge coefficients minimize a penalized residual sum of squares.

    Here 'alpha' >= 0 is a complexity parameter that controls the amount of shrinkage.
    the larger the value of 'alpha' the greater the amount of shrinkage. This makes the coefficients more robust to
    collinearity.
    """
    def __init__(self, X, y):
        super(PiRidgeRegression, self).__init__(X=X, y=y)

    def fit(self, fit_intercept=True, alpha=1.):
        """

        :param alpha: regularization strength. must be a positive <float>. Improves the condition of the problem and
                      reduces the variance of the estimates. Larger values specify stronger regularization.
        """
        super().fit(fit_intercept=fit_intercept)
        self.coefficients = np.linalg.inv(self.X.T.dot(self.X)
                                          +
                                          np.identity(self.n_features).dot(alpha)).dot(self.X.T.dot(self.y))

    def predict(self, X):
        return super().predict(X=X)
