import numpy as np
import math


def k_epanechnikov(x):
        """
        Epanechnikov Kernel
        """
        if isinstance(x, np.ndarray):
            x = np.sqrt(np.sum(x.T.dot(x)))
        if abs(x) > 1:
            return 0
        else:
            return 3 / 4 * (1 - x**2)


def k_gaussian(x):
    """
    Gaussian Kernel
    """
    if isinstance(x, np.ndarray):
        x = np.sqrt(np.sum(x.T.dot(x)))

    return 1 / (np.sqrt(2 * math.pi)) * np.exp( - (x**2) / 2 )


def k_bicube(x):
    """
    Bicube Kernel
    """
    if isinstance(x, np.ndarray):
        x = np.sqrt(np.sum(x.T.dot(x)))

    if abs(x) < 1:
        return 15/16 * (1 - abs(x)**2 )**2
    else:
        return 0


def k_triangular(x):
    """
    Triangular Kernel
    """
    if isinstance(x, np.ndarray):
        x = np.sqrt(np.sum(x.T.dot(x)))

    if abs(x) < 1:
        return 1 - abs(x)
    else:
        return 0


def k_tricube(x):
    """
    Tricube Kernel
    """
    if isinstance(x, np.ndarray):
        x = np.sqrt(np.sum(x.T.dot(x)))

    if abs(x) < 1:
        return 70/80 * (1-abs(x)**3)**3
    else:
        return 0
