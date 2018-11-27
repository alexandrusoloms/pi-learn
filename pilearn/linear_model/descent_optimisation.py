import numpy as np
from ..base import PiBaseRegression


class PiGradientDescent(PiBaseRegression):

    def __init__(self, alpha, iterations, normalise=False, lambda_=0.0):
        super(PiGradientDescent, self).__init__()
        self.alpha = alpha
        self.iterations = iterations
        self.normalise = normalise
        self.l = lambda_
        self.fit_intercept = None
        self.coefficients = None
        self.cost_vector = None

    @staticmethod
    def calculate_hypothesis(X, coefficients, index):
        return np.c_[coefficients].T.dot(X[index].reshape(-1, 1))[0]

    def compute_cost_regularized(self):
        """
        computes the regularized cost using MSE.
        :return:
        """
        # computes regularized cost for regularized linear regression
        # takes as an input X of training examples, a parameter vector- theta
        # and lambada- l and an output vector- y

        theta_squared_error = .0
        for i in range(self.n_samples):
            hypothesis = self.calculate_hypothesis(self.X, self.coefficients, i)
            output = self.y[i]
            squared_error = (hypothesis - output) ** 2
            theta_squared_error = theta_squared_error + squared_error

        total_regularised_error = .0
        for i in range(len(self.coefficients)):
            total_regularised_error = total_regularised_error + self.coefficients[i] ** 2

        total_regularised_error = total_regularised_error * self.l
        return (1 / (2 * self.n_samples)) * (theta_squared_error + total_regularised_error)

    def fit(self, X, y, fit_intercept=True):
        """

        :param X:
        :param y:
        :param fit_intercept:
        :return:
        """
        super().fit(X=X, y=y, fit_intercept=fit_intercept)
        n_samples = self.n_samples
        self.coefficients = np.random.randn(self.n_features, 1)
        self.cost_vector = list()
        for it in range(self.iterations):
            temporary_coefficients_placeholder = list()
            for ind, coefficients_i in enumerate(self.coefficients):
                sigma = 0
                for i in range(n_samples):
                    hypothesis = self.calculate_hypothesis(self.X, self.coefficients, i)
                    output = self.y[i]
                    sigma = sigma + (hypothesis - output) * self.X[:, ind][i]
                if ind == 0:  # coefficients 0
                    coefficients_i = coefficients_i - (self.alpha / self.n_samples) * sigma
                else:
                    coefficients_i = coefficients_i * (1 - self.alpha * (self.l / self.n_samples)) - \
                                     (self.alpha / self.n_samples) * sigma
                temporary_coefficients_placeholder.append(coefficients_i)
            self.coefficients = np.c_[temporary_coefficients_placeholder]

            self.cost_vector.append(self.compute_cost_regularized())

    def predict(self, X):
        return super().predict(X=X)
