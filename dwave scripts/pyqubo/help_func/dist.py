import numpy as np


def dist(i, j, cities):

    pos_i = cities[i]
    pos_j = cities[j]

    return np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2)
