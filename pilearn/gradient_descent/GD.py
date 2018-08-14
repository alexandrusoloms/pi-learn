import numpy as np
from base import PiBaseRegression

class PiGradientDescent(PiBaseRegression):

    def __init__(self, X, y):
        super(PiGradientDescent, self).__init__(X=X, y=y)

    def fit(self, learning_rate=.01, num_iters=10000):
        """
           Performs gradient descent to learn theta
        """
        m = self.y.size  # number of training examples
        theta = np.random.randn(self.n_features, 1)
        for i in range(num_iters):
            gradient = np.array(2 / m).dot(self.X.T.dot(self.X.dot(theta) - self.y))
            theta = theta - learning_rate * gradient
        self.coefficients = theta

    def predict(self, X):
        super().predict()
        return X.dot(self.coefficients)

class PiStochasticGradientDescent(PiBaseRegression):

    def __init__(self, X, y):
        super(PiStochasticGradientDescent, self).__init__(X=X, y=y)

    @staticmethod
    def _learning_schedule(t):
        t0, t1 = 5, 50
        return t0 / (t + t1)

    def fit(self, learning_rate=.01, num_iters=50):
        """
           Arguments:
           f -- the function to optimize, it takes a single argument
                and yield two outputs, a cost and the gradient
                with respect to the arguments
           theta0 -- the initial point to start SGD from
           num_iters -- total iterations to run SGD for
           Return:
           theta -- the parameter value after SGD finishes
        """
        n_epochs = 50
        theta = np.random.randn(self.n_features, 1)
        m = self.X.shape[0]
        for epoch in range(num_iters):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = self.X[random_index: random_index + 1]
                yi = self.y[random_index: random_index + 1]
                gradient = np.array(2).dot(xi.T.dot(xi.dot(theta) - yi))
                eta = self._learning_schedule(epoch * m + i)
                theta = theta - eta * gradient
        self.coefficients = theta

    def predict(self, X):
        super().predict()
        return X.dot(self.coefficients)
