from typing import List

from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import numpy as np
import numpy as np
from sklearn.decomposition import PCA
from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.plotting import plot_3d

from data.dataLoader import loadAllDataAsArrays
from datapoints import pointsAsArray, datapoint
from helpers.seabornTools import scatterPlot3D
from paths import DataDir


def computeNearestNeighboursMatrix(X, k=7, debug=False):  # (x,y,z) , (thickness tc.)

    kdt = KDTree(X, metric='euclidean')
    (dists, indices) = kdt.query(X, k=k, return_distance=True)

    if debug:
        N = len(X)
        for i in range(N):
            print(dists[i])
    return dists, indices


def fMinLeastSquare(data, initialGuess):
    # distance = (plane_xyz * X.T).sum(axis=1) + initialGuess[:-1]
    pass


def fitPointsToPlane(data, mode="LSE"):
    '''
    The general description of a plane is: Ax  + by = z
    Applying naively SVD yields a best fit towards the z axis in above's case,
    so the approximations closer to the dent might become increasingly inprecise.
    :param data:
    :return:
    '''
    if mode == "LSE":
        from scipy.optimize import leastsq
        initialGuess = 0
        sol = leastsq(fMinLeastSquare, initialGuess, args=(None, data))[0]
        pass
    elif mode == "PCA":
        data = np.array(data)

        m = np.mean(data, axis=0)
        pca = PCA(n_components=3)
        pca.fit(data)

        ev = pca.components_

    base = m
    v1 = ev[0]
    v2 = ev[1]

    return base, v1, v2


def estimatePlaneThroughData(data):
    data = np.array(data)

    m = np.mean(data, axis=0)

    pca = PCA(n_components=3)
    pca.fit(data)

    ev = pca.components_

    plane = Plane.from_vectors(m, ev[0], ev[1])
    return plane


def constructTransformation(plane):
    cartesian = plane.cartesian()

    a = cartesian[0]
    b = cartesian[1]
    c = cartesian[2]
    d = cartesian[3]

    D = np.sqrt(a ** 2 + b ** 2 + c ** 2)

    cosTheta = c / D
    sinTheta = np.sqrt(a ** 2 + b ** 2) / D
    u1 = b / D
    u2 = -a / D

    rotMatrix = [[cosTheta + u1 ** 2 * (1 - cosTheta), u1 * u2 * (1 - cosTheta), u2 * sinTheta],
                 [u1 * u2 * (1 - cosTheta), cosTheta + u2 ** 2 * (1 - cosTheta), -u1 * sinTheta],
                 [-u2 * sinTheta, u1 * sinTheta, cosTheta]]
    rotMatrix = np.array(rotMatrix)
    return rotMatrix


def alignData(data):
    plane = estimatePlaneThroughData(data)
    a, b, c, d = plane.cartesian()
    rotMatrix = constructTransformation(plane)

    if np.abs(c) > 0:
        N = len(data)
        for i in range(N):
            data[i, 2] = data[i, 2] - d / c

    rotated = data @ rotMatrix.T
    projected = rotated[:, :2]
    translated = projected - projected[0]
    return translated  # aligned data


def isFarleft(x):
    ret = x < -0.75
    return ret


def isLeft(x):
    ret = x < -0.25 and x > -0.75
    return ret


def isCenter(x):
    ret = x > -0.25 and x < 0.25
    return ret


def isRight(x):
    ret = x > 0.25 and x < 0.75
    return ret


def isFarright(x):
    ret = x > 0.75
    return ret


def getLabels(newData, indices, k=7):
    farleft = 0
    left = 0
    center = 0
    right = 0
    farright = 0

    "[-1, -0.4, 0, 0.4, 1]"
    d = dict()  # associations
    for p, i in zip(newData,
                    indices):  # can likely be further parallelised, e.g. via np.where and and np.logical_or etc.
        x = p[0]
        if isCenter(x):
            center = center + 1
            d[str(i)] = "C"
        elif isRight(x):
            right = right + 1
            d[str(i)] = "R"
        elif isLeft(x):
            left = left + 1
            d[str(i)] = "L"
        elif isFarright(x):
            farright = farright + 1
            d[str(i)] = "FR"
        elif isFarleft(x):
            farleft = farleft + 1
            d[str(i)] = "FL"
        else:
            assert False
    assert farleft + left + center + right + farright == k
    return d


def getAffiliation(curIndices, curDists):
    cp = curIndices[0]
    N = len(curIndices)

    for i in range(1, N):
        cur_x = curDists[0]


def inFirstSet(x):
    ret = isCenter(x) or isFarleft(x) or isFarright(x)
    return ret


def visuazlizePointMap(data):
    scatterPlot3D(data, dst=".", title="something.png", colours='b', alpha=0.5)


def curSet(v1, v2, m):
    ret = None
    if m == 0:
        ret = v1
    elif m == 1:
        ret = v2
    else:
        assert False


def oppSet(v1, v2, m):
    ret = None
    if m == 0:
        ret = v2
    elif m == 1:
        ret = v1
    else:
        assert False


def detSwitch(v1, v2):
    m = None
    if len(v1) != 0:
        m = 0
    else:
        if len(v2) != 0:
            m = 1
        else:
            m = -1
    return m


def terminateLoop():
    return True


def constructMatrix(data, k=7):
    N = len(data)

    visited1 = set()
    visited2 = set()

    next1 = set()
    next2 = set()

    dists, indices = computeNearestNeighboursMatrix(data, k=k)

    sn = np.random.randint(N)  # start index
    next1.add(sn)

    pointMap = np.zeros((N, 2))
    pointMap[sn, 0] = 1000
    pointMap[sn, 1] = 1000
    ''''
    The grid is encoded inside the pointMap. 
        Even indices relate to next1
        Odd indices relate to next2
    '''

    # Crude way to avoid an endless loop
    iterationsMax = 3000
    iterationsCur = 0

    switch = 0
    ''''
    This flag determines if a point p belongs to either set next1 or next2.
    By definition: 
        p==0 means p belongs to next1, 
        p==1 means p belongs to next2
    '''

    while len(next1) != 0 or len(next2) != 0:
        if iterationsCur > iterationsMax:
            break

        switch = detSwitch(next1, next2)
        if switch == -1:
            break

        curS = curSet(next1, next2, switch)
        oppS = oppSet(next1, next2, switch)
        curV = curSet(visited1, visited2, switch)
        oppV = oppSet(visited1, visited2, switch)

        # collect neighbours
        if len(curS) != 0:

            curInd = curS.pop()
            visited1.add(curInd)

            curIndices = indices[curInd]
            alignedData = alignData(data[curIndices])

            flps = list()  # far left points
            cps = list()  # center points
            frps = list()  # far right points

            M = len(curIndices)
            for j in range(M):
                nbInd = curIndices[j]
                if nbInd in curV:
                    continue
                else:
                    lx = alignedData[j][0]

                    # determine affiliation
                    if isFarleft(lx):
                        flps.append(nbInd)
                    elif isCenter(lx):
                        cps.append(nbInd)
                    elif isFarright(lx):
                        frps.append(nbInd)
                    else:
                        continue

            flps = np.array(flps)  # far left points
            cps = np.array(cps)  # center points
            frps = np.array(frps)  # far right points

            # update pointMap
            # get relative coordinates
            curIndX = pointMap[curInd, 0]
            curIndY = pointMap[curInd, 1]

            # update leftPoint
            M = len(flps)
            if M > 0:
                flpDists = np.zeros((M))
                for j in range(M):
                    flp = flps[j]
                    dist = np.abs(data[flp][0] - data[curInd][0])
                    flpDists[j] = dist
                lNb = np.argmin(flps, axis=0)  # left neighbour
                pointMap[flps[lNb], 0] = curIndX - 1
                pointMap[flps[lNb], 1] = curIndY - 1
                next1.add(flps[lNb])

            M = len(frps)
            if M > 0:
                frpDists = np.zeros((M))
                for j in range(M):
                    frp = frps[j]
                    dist = np.abs(data[frp][0] - data[curInd][0])
                    frpDists[j] = dist
                rNb = np.argmin(frpDists)  # right neighbour
                pointMap[frps[rNb], 0] = curIndX + 1
                pointMap[frps[rNb], 1] = curIndY + 1
                next1.add(flps[rNb])

            M = len(cps)

            if M > 0:
                abovePoints = np.zeros((M, 2))
                belowPoints = np.zeros((M, 2))
                numAbove = 0
                numBelow = 0
                for j in range(M):
                    cp = cps[j]
                    diff = data[curInd][1] - data[cp][1]
                    if diff < 0:
                        abovePoints[numAbove, 1] = diff
                        abovePoints[numAbove, 0] = cp
                        numAbove = numAbove + 1
                    elif diff > 0:
                        belowPoints[numAbove, 1] = diff
                        belowPoints[numAbove, 0] = cp
                        numBelow = numBelow + 1
                    else:
                        assert False
                    next1.add(cp)

                abovePoints = abovePoints[:numAbove, :]
                belowPoints = belowPoints[:numBelow, :]

                if len(abovePoints) != 0:
                    topNeighbour = abovePoints[np.argmin(abovePoints[:, 0]), 0]  # top neighbour
                    topNeighbour = int(topNeighbour)
                    pointMap[topNeighbour, 0] = int(curIndX)
                    pointMap[topNeighbour, 1] = int(curIndY + 1)

                if len(belowPoints) != 0:
                    bottomNeighbour = belowPoints[np.argmin(belowPoints[:, 0]), 0]  # bottom neighbour
                    bottomNeighbour = int(bottomNeighbour)
                    pointMap[bottomNeighbour, 0] = int(curIndX)
                    pointMap[bottomNeighbour, 1] = int(curIndY - 1)
        iterationsCur = iterationsCur + 1

    for point in pointMap:
        print(point)
    print("Are we done?")

    ####
    ass = np.array(list(visited1))
    visitedData = data[ass]
    visuazlizePointMap(visitedData)


def test():
    src = DataDir.cleanSamples
    datas = loadAllDataAsArrays(src=src, normalizeData=False)
    data = datas[0]

    constructMatrix(data)


test()