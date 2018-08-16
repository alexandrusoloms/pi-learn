#import LinearRegression
#from LinearRegression import PiLinearRegression
from .LinearRegression import PiLinearRegression
from .GD import PiGradientDescent, PiStochasticGradientDescent
from .ridge import PiRidgeRegression
__all__ = [
    'PiLinearRegression',
    'PiBaseRegression',
    'PiRidgeRegression',
    'PiGradientDescent',
    'PiStochasticGradientDescent'
            ]
