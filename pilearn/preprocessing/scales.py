import numpy as np


def min_max_scaler(X):
    """
    rescale between 0, 1
    """
    list_of_arrays = list()
    n_samples, n_features = X.shape
    for i in X.T:
        if len(set(i)) == n_samples:
            normed_val = [(j - min(i)) / (max(i) - min(i)) for j in i]
            list_of_arrays.append(normed_val)
        else:
            list_of_arrays.append(i)
    return np.c_[list_of_arrays].T


def mean_variance_scaler(X):
    """
    reduce to mean 0, std 1.
    """
    list_of_arrays = list()
    n_samples, n_features = X.shape
    for i in X.T:
        if len(set(i)) == n_samples:
            normed_val = [(j - np.mean(i)) / np.std(i) for j in i]
            list_of_arrays.append(normed_val)
        else:
            list_of_arrays.append(i)
    return np.c_[list_of_arrays].T

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
