import math


def eucledian_distance(vector_1, vector_2):
    distance = 0
    for ind, x in enumerate(range(len(vector_1))):
        distance += np.square(x - vector_2[ind])
    return np.sqrt(distance)
