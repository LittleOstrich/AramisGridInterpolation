import numpy as np
import pandas as pd

from helpers.csvTools import writeDataframeToXlsx, appendDictsToDf


class interpolationModes:
    default = "default"


def normalizePointMap(pointMap):
    hm = np.min(pointMap[:, 0], axis=0)
    wm = np.min(pointMap[:, 1], axis=0)

    pointMap[:, 0] = pointMap[:, 0] - hm
    pointMap[:, 1] = pointMap[:, 1] - wm
    return pointMap


def pointMapToCoordinateMap(pointMap):
    hM = np.max(pointMap[:, 0], axis=0)
    hm = np.min(pointMap[:, 0], axis=0)
    wM = np.max(pointMap[:, 1], axis=0)
    wm = np.min(pointMap[:, 1], axis=0)

    h = int(hM - hm)
    w = int(wM - wm)
    pointMap[:, 0] = pointMap[:, 0] - hm
    pointMap[:, 1] = pointMap[:, 1] - wm
    coordinateMap = np.zeros((h + 1, w + 1), dtype=int)

    N = len(pointMap)
    for i in range(N):
        p = pointMap[i]
        coordinateMap[p[0], p[1]] = i
    return coordinateMap


def interpolate(originalData, pointMap, dstDir, mode=interpolationModes.default):
    oldPoints = list()
    newPoints = list()

    pointMap = normalizePointMap(pointMap)
    coordMap = pointMapToCoordinateMap(pointMap)

    keys = originalData[0].keys()

    newPointsDf = pd.DataFrame(columns=keys)
    oldPointsDf = pd.DataFrame(columns=keys)
    N = len(pointMap)
    for i in range(N):
        p = pointMap[i]
        x = p[0]
        y = p[1]
        if x % 2 == 0 and y % 2 == 0:
            if x + 1 >= len(coordMap) or y + 1 >= len(coordMap[0]) or y - 1 < 0:
                continue
            tN = coordMap[x + 1, y + 1]
            bN = coordMap[x + 1, y - 1]
            if tN != 0 and bN != 0:
                newPoint = dict()
                for key in keys:
                    val = (originalData[tN][key] + originalData[bN][key]) / 2
                    newPoint[key] = val
                    newPoint["id"] = len(newPoints)

                newPoints.append(newPoint)
                oldPoints.append(originalData[i])

    oldPointsDf = appendDictsToDf(oldPoints, oldPointsDf)
    newPointsDf = appendDictsToDf(newPoints, newPointsDf)
    writeDataframeToXlsx(oldPointsDf, dstDir, "oldPoints.xlsx")
    writeDataframeToXlsx(newPointsDf, dstDir, "newPoints.xlsx")

    plt.figure(1, figsize=(9, 4))

    # Specify a RGB color from float 3-tuple
    plt.subplot(1, 2, 1)
    plt.plot(x1, y1, '*', c=np.random.rand(3))
    plt.ylabel('RGB color from float 3-tuple')

    # Specify a color from a colormap from a single float
    plt.subplot(1, 2, 2)
    plt.plot(x1, y1, '*', c=plt.cm.viridis(np.random.rand()))
    plt.plot(x2, y2, 'o', c=plt.cm.viridis(np.random.rand()))
    plt.ylabel('Color from colormap from single float')

    plt.tight_layout()
    plt.show()
