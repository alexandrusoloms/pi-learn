from ..base import PiBaseNonParametricRegression
import numpy as np
from .kernels import K_bicube, K_Epanechnikov, K_Gaussian, K_triangular, K_tricube


class PiNonParametericRegression(PiBaseNonParametricRegression):
    def __init__(self, X, y, bandwidth, kernel='E'):
        super(PiNonParametericRegression, self).__init__(X=X, y=y)
        self.bandwith = bandwith
        self.kernel = kernel

    def m(cls, xi, kernel):
        """
        local fit regression

        :param xi: the x value
        :param kernel: a Choice arg
        :param h: bandwidth

        """
        y_list = []
        x_list = []

        for i in range(len(cls.X)):
            sub_m = (cls.X[i] - xi) / self.bandwith
            y_list.append(kernel(sub_m) * cls.y[i])
            x_list.append(kernel(sub_m))

        return np.sum(y_list) / np.sum(x_list)

    def predict():
