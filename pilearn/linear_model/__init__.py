#import LinearRegression
#from LinearRegression import PiLinearRegression
from .linear_regression import PiLinearRegression
from .gradient_descent import PiGradientDescent, PiStochasticGradientDescent
from .ridge import PiRidgeRegression
__all__ = [
    'PiLinearRegression',
    'PiBaseRegression',
    'PiRidgeRegression',
    'PiGradientDescent',
    'PiStochasticGradientDescent'
            ]
