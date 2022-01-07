import os.path

from PIL import Image
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
        curShape = np.asarray(arg.shape)
        if len(curShape) >= hp + 1:
            if curShape[hp] == desShape:
                arrays.append(arg)
    concatedArray = np.concatenate(arrays, axis=axis)
    return concatedArray


def matrixAsImage(A, show=False, save=True, dstDir=None, fn=None):
    img = Image.fromarray(A, "L")

    if show:
        img.show()
    if save:
        assert os.path.exists(dstDir)
        assert fn is not None
        dst = dstDir + os.sep + fn
        img.save(dst, "png")
    return img
