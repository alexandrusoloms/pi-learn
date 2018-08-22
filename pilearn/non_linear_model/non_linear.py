from ..base import PiBaseNonParametricRegression
import numpy as np
from .kernels import K_bicube, K_Epanechnikov, K_Gaussian, K_triangular, K_tricube


class PiNonParametericRegression(PiBaseNonParametricRegression):
    def __init__(self, X, y, bandwidth, kernel='E'):
        super(PiNonParametericRegression, self).__init__(X=X, y=y)
        self.bandwidth = bandwidth
        self.kernel = kernel

    @staticmethod
    def __epanechnikov(x):
        # Epanechnikov Kernel
        if isinstance(x, np.ndarray):
            x = np.sqrt(np.sum(x.T.dot(x)))
        if abs(x) > 1:
            return 0
        else:
            return 3 / 4 * (1 - x**2)

    def predict(self, X):
        fx_list = list()
        for b in X:
            o = [self.__epanechnikov(i) for i in
                [[(b- xi)/self.bandwidth][0]  for xi in X]]

            O = np.eye(self.n_samples) * np.array(o).reshape(-1, 1)

            f_x = O.dot(X).dot(np.linalg.inv(X.T.dot(O).dot(X))).dot(b).T.dot(self.y)
            # b.dot(np.linalg.inv(X.T.dot(O).dot(X))).dot(X.T).dot(O).dot(y)
            # O.dot(X).dot(np.linalg.inv(X.T.dot(O).dot(X))).dot(b).T.dot(self.y)
            #
            fx_list.append(f_x.tolist())
        return fx_list
