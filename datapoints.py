from typing import List

import numpy as np


def computeDist(p1, p2):
    dist = np.sqrt((p1.x - p2.x) ** 2 +
                   (p1.y - p2.y) ** 2 +
                   (p1.z - p2.z) ** 2)
    return dist


def pointsAsArray(points):
    N = len(points)

    X = np.zeros_like((N, 3))
    for i in range(N):
        p = points[i]
        X[i, 0] = p.x
        X[i, 1] = p.y
        X[i, 2] = p.z
    return X


class datapoint:

    def __init__(self, x, y, z, originalIndex):
        self.x = x
        self.y = y
        self.z = z
        self.originalIndex = originalIndex

        # relativePositions of the 4 nearest points given by originalIndex
        self.top = None
        self.bottom = None
        self.left = None
        self.right = None

        self.neighboursOriginalIndices = [-1, -1, -1, -1]
        self.neighboursDistances = [-1, -1, -1, -1]

        self.selectionCounter = None

    def setRelativPositions(self, points, globalAverageDist):

        N = len(points)
        for i in range(N):
            p: datapoint = points[i]
            dist = computeDist(self, p)

            isnotAnomaly = dist * 0.9 < globalAverageDist
            if isnotAnomaly:
                xDiff = np.abs(p.x - self.x)
                yDiff = np.abs(p.y - self.y)

                if xDiff > yDiff:
                    if p.x > self.x:
                        self.right = p.x
                    elif p.x < self.x:
                        self.left = p.x
                else:
                    if p.y > self.y:
                        self.bottom = p.y
                    elif p.y < self.y:
                        self.top = p.y
            else:
                print("Anomaly detected! PointId: ", self.originalIndex)


pass
