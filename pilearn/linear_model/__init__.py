from .linear_regression import PiLinearRegression
# from .gradient_descent import PiGradientDescent, PiStochasticGradientDescent
from .ridge import PiRidgeRegression
from .descent_optimisation import PiGradientDescent

__all__ = [
    'PiLinearRegression',
    'PiRidgeRegression',
    'PiGradientDescent',
    # 'PiStochasticGradientDescent'
            ]
