import numpy as np

from helpers.numpyTools import matrixAsImage


def centerPointMap(pointMap):
    miny = np.min(pointMap[:, 0])
    minx = np.min(pointMap[:, 1])

    pointMap[:, 0] = pointMap[:, 0] - miny
    pointMap[:, 1] = pointMap[:, 1] - minx
    pointMap = np.array(pointMap, dtype=int)
    return pointMap


def centeredPointMapToMatrix(pointMap):
    h = int(np.max(pointMap[:, 0]))
    w = int(np.max(pointMap[:, 1]))
    newPointMap = np.zeros((h, w))

    for p in pointMap:
        newPointMap[p[0] - 1, p[1] - 1] = 1
    return newPointMap


def pointMapToMatrix(pointMap):
    pointMap = centerPointMap(pointMap)
    newPointMap = centeredPointMapToMatrix(pointMap)
    return newPointMap


def percentilePointMap(pointMap, perVals):
    condYs = pointMap[:, 0] < perVals[0]
    condXs = pointMap[:, 1] < perVals[1]

    inds = list()
    for i in range(len(condXs)):
        if condXs[i] and condYs[i]:
            inds.append(i)
    inds = np.array(inds)

    percentiledPointMap = pointMap[inds, :]
    return percentiledPointMap


def storeVisualizedPointMap(pointMap, percentile, tS=20, save=True, show=False, dstDir=None, fn=None):
    centeredPointMap = centerPointMap(pointMap)
    perVals = np.percentile(centeredPointMap, percentile, axis=0)
    percentiledPointMap = percentilePointMap(centeredPointMap, perVals)
    pointMapMat = centeredPointMapToMatrix(percentiledPointMap)
    tiledPointMapMat = np.kron(pointMapMat, np.ones((tS, tS))) * 255
    m = matrixAsImage(tiledPointMapMat, show=show, save=save, dstDir=dstDir,
                      fn=fn)
    return m
