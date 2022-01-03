import numpy as np


def safeIndexing(arr, ind):
    if len(ind) == 0:
        return np.array([])
    else:
        return arr[ind]


def saveConcate(args, axis=0):
    hp = [_ for _ in range(len(args[0].shape))]
    hp.remove(axis)
    hp = np.array(hp)

    desShape = np.asarray(args[0].shape)
    desShape = desShape[hp]

    arrays = list()
    for arg in args:
        if np.asarray(arg.shape)[hp] == desShape:
            arrays.append(arg)
    concatedArray = np.concatenate(arrays, axis=axis)
    return concatedArray
