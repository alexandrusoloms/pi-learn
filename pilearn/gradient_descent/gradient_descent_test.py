import sys
sys.path.append('../')
sys.path.append('')
from GD import PiStochasticGradientDescent
import numpy as np
from base import PiBaseRegression

X = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.array(list(4.5 + x*10 + np.random.poisson() for x in X))
X = np.c_[np.ones(len(X)), X]

sgd = PiStochasticGradientDescent(X=X, y=y)
sgd.fit()
y_pred = (sgd.predict(X=X))

for ind, i in enumerate(y_pred):
    print('y_pred -->{}, real--->{}'.format(i, y[ind])) 
