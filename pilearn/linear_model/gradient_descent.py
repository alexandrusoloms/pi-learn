import sys
import numpy as np
from ..base import PiBaseRegression


class PiGradientDescent(PiBaseRegression):

    def __init__(self, X, y):
        super(PiGradientDescent, self).__init__(X=X, y=y)

    def fit(self, learning_rate=.01, num_iters=10000):
        """
           Performs gradient descent to learn theta
        """
        n_samples = self.n_samples
        theta = np.random.randn(self.n_features, 1)
        for i in range(num_iters):
            gradient = np.array(2 / n_samples).dot(self.X.T.dot(self.X.dot(theta) - self.y))
            theta = theta - learning_rate * gradient
        self.coefficients = theta

    def predict(self, X):
        return super().predict(X=X)


class PiStochasticGradientDescent(PiBaseRegression):

    def __init__(self, X, y):
        super(PiStochasticGradientDescent, self).__init__(X=X, y=y)

    @staticmethod
    def _learning_rate_function(t):
        """
        this will change the learning rate with every iteration
        """
        t0, t1 = 5, 50
        return t0 / (t + t1)

    def fit(self, num_iters=10000):

        theta = np.random.randn(self.n_features, 1)
        n_samples = self.n_samples
        for i in range(num_iters):
            for j in range(n_samples):
                g_index = np.random.randint(n_samples)
                xi = self.X[g_index: g_index + 1]
                yi = self.y[g_index: g_index + 1]
                gradient = np.array(2).dot(xi.T.dot(xi.dot(theta) - yi))
                eta = self._learning_rate_function(i * n_samples + j)
                theta = theta - eta * gradient
        self.coefficients = theta




    def predict(self, X):
        return super().predict(X=X)
