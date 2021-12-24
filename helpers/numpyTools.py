import numpy as np


def safeIndexing(arr, ind):
    if len(ind) == 0:
        return np.array([])
    else:
        return arr[ind]
