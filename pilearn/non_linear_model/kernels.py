import numpy as np

def K_Epanechnikov(x):
        """
        Epanechnikov Kernel
        """
        if abs(x) < np.sqrt(5):
            return 1/np.sqrt(5) * 3/4 * (1 - 1/5 * x**2)
        else:
            return 0


def K_Gaussian(x):
    """
    Gaussian Kernel
    """
    return 1 / (np.sqrt(2 * math.pi)) * np.exp( - (x**2) / 2 )


def K_bicube(x):
    """
    Bicube Kernel
    """
    if abs(x) < 1:
        return 15/16 * (1 - abs(x)**2 )**2
    else:
        return 0

def K_triangular(x):
    """
    Triangular Kernel
    """
    if abs(x) < 1:
        return 1 - abs(x)
    else:
        return 0

def K_tricube(x):
    """
    Tricube Kernel
    """
    if abs(x) < 1:
        return 70/80 * (1-abs(x)**3)**3
    else:
        return 0
