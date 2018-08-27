from ..base import PiBaseNonParametricRegression
import numpy as np
from .kernels import k_bicube, k_epanechnikov, k_gaussian, k_triangular, k_tricube

kernels_dict = {
    'E': k_epanechnikov,
    'G': k_gaussian,
    'T': k_tricube,
    'B': k_bicube,
    'TC': k_tricube
}


class PiNonParametericRegression(PiBaseNonParametricRegression):
    def __init__(self, X, y, bandwidth, kernel='E'):
        """
        select a kernel as follows:
         {
             'E' for  k_epanechnikov,
             'G' for k_gaussian,
             'T' for  k_tricube
             'B' for k_bicube,
             'TC' for k_tricube
         }
        """
        super(PiNonParametericRegression, self).__init__(X=X, y=y)
        self.bandwidth = bandwidth
        self.kernel = kernels_dict[kernel]

    @staticmethod
    def __epanechnikov(x):
        # Epanechnikov Kernel
        if isinstance(x, np.ndarray):
            x = np.sqrt(np.sum(x.T.dot(x)))
        if abs(x) > 1:
            return 0
        else:
            return 3 / 4 * (1 - x**2)

    def fit_predict(self, X):
        fx_list = list()
        for b in X:
            o = [self.kernel(i) for i in
                [[(b- xi)/self.bandwidth][0]  for xi in X]]

            O = np.eye(self.n_samples) * np.array(o).reshape(-1, 1)

            f_x = O.dot(X).dot(np.linalg.inv(X.T.dot(O).dot(X))).dot(b).T.dot(self.y)
            # b.dot(np.linalg.inv(X.T.dot(O).dot(X))).dot(X.T).dot(O).dot(y)
            # O.dot(X).dot(np.linalg.inv(X.T.dot(O).dot(X))).dot(b).T.dot(self.y)
            #
            fx_list.append(f_x.tolist())
        return fx_list
