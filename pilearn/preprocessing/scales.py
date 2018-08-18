import numpy as np
from .base import PiBasePreprocessing


class PiMinMaxScaler(PiBasePreprocessing):
    """
    ``PiMinMaxScaler`` inherits the abstractmethod ``PiBaseRegression``
    """
    def __init__(self, X):
        super(PiMinMaxScaler, self).__init__(X=X)

    def fit(self):
        for i in self.X.T:
            normed_val = [(j - min(i)) / (max(i) - min(i)) for j in i]
            self.list_of_arrays.append(normed_val)

    def transform(self):
        self.transformed = np.c_[self.list_of_arrays].T

    def fit_transform(self):
        self.fit()
        self.transform()
        return self.transformed

class PiMeanVarianceScaler(PiBasePreprocessing):
    """
    <to add>
    """
    def __init__(self, X):
        super(PiMeanVarianceScaler, self).__init__(X=X)

    def fit(self):
        for i in self.X.T:
            normed_val = [(j - np.mean(i)) / np.std(i) for j in i]
            self.list_of_arrays.append(normed_val)

    def transform(self):
        self.transformed = np.c_[self.list_of_arrays].T

    def fit_transform(self):
        self.fit()
        self.transform()
        return self.transformed

def make_one_hot(X):
    """
    creates a one hot array
    """
    list_of_vals = list()
    for i in X.T:
        set_of_categories = list(set(i))
        for cat in set_of_categories:
            list_of_vals.append([int(j) for j in (cat == i)])
    return np.array(list(zip(*list_of_vals)))
