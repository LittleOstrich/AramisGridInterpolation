from typing import List

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import numpy as np
import numpy as np
from sklearn.decomposition import PCA
from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.plotting import plot_3d

from data.dataKeys import interpolatedDataHeaders
from data.dataLoader import loadAllDataAsArrays
from datapoints import pointsAsArray, datapoint
from helpers.csvTools import writeCsv
from helpers.seabornTools import scatterPlot3D
from helpers.timeTools import myTimer
from paths import DataDir


def computeNearestNeighboursMatrix(X, k=7, debug=False):  # (x,y,z) , (thickness tc.)

    kdt = KDTree(X, metric='euclidean')
    (dists, indices) = kdt.query(X, k=k, return_distance=True)

    if debug:
        N = len(X)
        for i in range(N):
            print(dists[i])
    return dists, indices


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
        initialGuess = np.array([0, 0, 0])
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
    """
    Cheap way to construct a plane through the data
    :param data:
    :return:
    """
    data = np.array(data)

    m = np.mean(data, axis=0)

    pca = PCA(n_components=3)
    pca.fit(data)

    ev = pca.components_

    plane = Plane.from_vectors(m, ev[0], ev[1])
    return plane


def constructTransformation(plane):
    """
    Rotate the plane to make it parallel to the xy- plane
    :param plane:
    :return:
    """
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
    """
    Data consists of a center point and certain number of nearest neighbours (7 be default).
    The center point is the first row entry in data.
    1. fit a plane to the data,
    2. Construction a rotation matrix to align the data (and the plane) so they are
    parallel to the xy plane
    3. center the data, so above-mentioned center point has the coordinates (0,0,0)
    The resulting patterns have very very characteristic shapes and one can easily define
    magic numbers that are most likely invariant across the dataset
    :param data:
    :return:
    """
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


# everyone likes magic numbers
def isFarleft(x):
    ret = x < -0.75
    return ret


# magic numbers
def isLeft(x):
    ret = x < -0.25 and x > -0.75
    return ret


# magic numbers
def isCenter(x):
    ret = x > -0.25 and x < 0.25
    return ret


# magic numbers
def isRight(x):
    ret = x > 0.25 and x < 0.75
    return ret


# magic numbers
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


def curSet(v1, v2, m):
    ret = None
    if m == 0:
        ret = v1
    elif m == 1:
        ret = v2
    else:
        assert False
    return ret.copy()


def oppSet(v1, v2, m):
    ret = None
    if m == 0:
        ret = v2
    elif m == 1:
        ret = v1
    else:
        assert False
    return ret.copy()


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


def determinePosition(alignedData, curIndices):
    flps = list()  # far left points
    lps = list()  # left points
    cps = list()  # center points
    rps = list()  # right points
    frps = list()  # far right points

    M = len(curIndices)
    for j in range(M):
        nbInd = curIndices[j]

        lx = alignedData[j][0]

        # determine affiliation
        if isFarleft(lx):
            flps.append(nbInd)
        elif isLeft(lx):
            lps.append(nbInd)
        elif isCenter(lx):
            cps.append(nbInd)
        elif isRight(lx):
            rps.append(nbInd)
        elif isFarright(lx):
            frps.append(nbInd)
        else:
            print(curIndices)
            print(alignedData)
            assert False

    flps = np.array(flps)  # far left points
    lps = np.array(lps)  # left points
    cps = np.array(cps)  # center points
    rps = np.array(rps)  # right points
    frps = np.array(frps)  # far right points

    return flps, lps, cps, rps, frps


def startNumber(sn, N):
    if sn is None:
        sn = np.random.randint(N)  # start index
    return sn


def constructMatrix(data, k=7, sn=None, iterationsMax=50000, hexagonsOnly=False, debug=False):
    """
    Base assumption: The data be separated into 2 grids (g1, g2) that are just shifted to another.
    Given a desired triangle in the mesh, 1 point belongs to the first grid, 2 point belong to the second grid
    (or vice versa)
    Steps:
    1. Determine for every point its nearest neighbours
    2. Randomly choose a starting point P, wlog this point belongs to g1 and has
    a suitable location in the pointmap e.g. 1000, 1000
    3. Determine the local shape around P consisting of its nearest neighboring points
    A, B, C...
    4. Determine if the points belong to either g1 or g2 via magic numbers
    5. Update the pointmap with the indices of the data - g1, g2 are encoded together in it.
    E.g. points in g1 only have even indices [i,j], g2 only odd indices [i,j].
    The rest of the values in the pointmap can be kept -1
    6. Crop the pointmap to a suitable size
    :param data:
    :param k:
    :param sn:
    :param iterationsMax:
    :param hexagonsOnly:
    :param debug:
    :return: pointmap
    """

    se = None
    if debug:
        se = myTimer("constructMatrixTimer")
    N = len(data)

    visited1 = set()
    visited2 = set()

    next1 = set()
    next2 = set()

    dists, indices = computeNearestNeighboursMatrix(data, k=k)

    sn = startNumber(sn, N)  # start index
    next1.add(sn)

    pointMap = np.zeros((N, 2), dtype=int)
    pointMap[sn, 0] = 1000
    pointMap[sn, 1] = 1000

    ''''
    The grid is encoded inside the pointMap. 
        Even indices relate to next1
        Odd indices relate to next2
    '''

    # Crude way to avoid an endless loop
    iterationsMax = iterationsMax  # sensible value is necessary, value has to big enough?
    iterationsCur = 0
    debugStepSize = 1000
    switch = 0

    ''''
    This flag determines if a point p belongs to either set next1 or next2.
    By definition: 
        p==0 means p belongs to next1, 
        p==1 means p belongs to next2
    '''
    while len(next1) != 0 or len(next2) != 0:
        if debug:
            if iterationsCur % debugStepSize == 0:
                se.start()
        if iterationsCur > iterationsMax:
            break

        switch = detSwitch(next1, next2)
        if switch == -1:
            break

        # we are continuously swapping between odd and even points
        curS = curSet(next1, next2, switch)
        oppS = oppSet(next1, next2, switch)
        curV = curSet(visited1, visited2, switch)
        oppV = oppSet(visited1, visited2, switch)

        # getting the next val for the current iteration
        curInd = curS.pop()
        curV.add(curInd)

        curIndices = indices[curInd]

        # fit a plane to the data and project the data onto it
        alignedData = alignData(data[curIndices])

        # determine the relative positions of the nearest neighbours
        flps, lps, cps, rps, frps = determinePosition(alignedData, curIndices)

        if hexagonsOnly:
            assert k == 7

            if len(flps) != 0 or len(frps) != 0:
                # iterationsCur = iterationsCur + 1
                next1 = curS.copy()
                next2 = oppS.copy()
                continue

        # update pointMap
        # get relative coordinates
        curIndX = pointMap[curInd, 0]
        curIndY = pointMap[curInd, 1]

        # update far left point
        M = len(flps)
        if M > 0:
            flpDists = np.zeros((M))
            for j in range(M):
                flp = flps[j]
                dist = np.abs(data[flp][0] - data[curInd][0])
                flpDists[j] = dist
            lNb = np.argmin(flps, axis=0)  # left neighbour
            pointMap[flps[lNb], 0] = curIndX - 2
            pointMap[flps[lNb], 1] = curIndY
            curS.add(flps[lNb])

        # updating left points
        M = len(lps)
        if M == 2:
            lp = lps[0]
            lq = lps[1]
            py = data[lp][1]
            qy = data[lq][1]
            cpy = data[curInd][1]
            if py < cpy and qy > cpy:
                pointMap[lp, 0] = curIndX - 1
                pointMap[lp, 1] = curIndY - 1
                pointMap[lq, 0] = curIndX - 1
                pointMap[lq, 1] = curIndY + 1
                oppS.add(lp)
                oppS.add(lq)
            elif py > cpy and qy < cpy:
                pointMap[lp, 0] = curIndX - 1
                pointMap[lp, 1] = curIndY + 1
                pointMap[lq, 0] = curIndX - 1
                pointMap[lq, 1] = curIndY - 1
                oppS.add(lp)
                oppS.add(lq)

        # updating right points
        M = len(rps)
        if M == 2:
            rp = rps[0]
            rq = rps[1]
            py = data[rp][1]
            qy = data[rq][1]
            cpy = data[curInd][1]
            if py < cpy and qy > cpy:
                pointMap[rp, 0] = curIndX + 1
                pointMap[rp, 1] = curIndY - 1
                pointMap[rq, 0] = curIndX + 1
                pointMap[rq, 1] = curIndY + 1
                oppS.add(rp)
                oppS.add(rq)
            elif py > cpy and qy < cpy:
                pointMap[rp, 0] = curIndX + 1
                pointMap[rp, 1] = curIndY + 1
                pointMap[rq, 0] = curIndX + 1
                pointMap[rq, 1] = curIndY - 1
                oppS.add(rp)
                oppS.add(rq)

        # set far right points
        M = len(frps)
        if M > 0:
            frpDists = np.zeros((M))
            for j in range(M):
                frp = frps[j]
                dist = np.abs(data[frp][0] - data[curInd][0])
                frpDists[j] = dist
            rNb = np.argmin(frpDists)  # far right neighbour
            pointMap[frps[rNb], 0] = curIndX + 2
            pointMap[frps[rNb], 1] = curIndY
            curS.add(frps[rNb])

        # order the points directly above/below the center point
        # creating a (index, distance) pair of arrays
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
                    continue
                curS.add(cp)

            abovePoints = abovePoints[:numAbove, :]
            belowPoints = belowPoints[:numBelow, :]

            if len(abovePoints) != 0:
                topNeighbour = abovePoints[np.argmin(abovePoints[:, 0]), 0]  # top neighbour
                topNeighbour = int(topNeighbour)
                pointMap[topNeighbour, 0] = int(curIndX)
                pointMap[topNeighbour, 1] = int(curIndY + 2)

            if len(belowPoints) != 0:
                bottomNeighbour = belowPoints[np.argmin(belowPoints[:, 0]), 0]  # bottom neighbour
                bottomNeighbour = int(bottomNeighbour)
                pointMap[bottomNeighbour, 0] = int(curIndX)
                pointMap[bottomNeighbour, 1] = int(curIndY - 2)

        if debug:
            if iterationsCur % debugStepSize == 0:
                se.end(showReport=False)
                print("iterationsCur: ", iterationsCur)
        iterationsCur = iterationsCur + 1

        # used a hard instead of weak copy to avoid conflicts between references
        visited1 = curV.copy()
        visited2 = oppV.copy()
        next1 = curS.copy()
        next2 = oppS.copy()

    print("Done with constructMatrix")
    return visited1, visited2, pointMap


def handlePointsToTheSide(data, indices, d, side="L"):
    spy1 = data[0][1]
    spy2 = data[1][1]

    # our center point in the hexagon coincides with origin of the coordinate system
    if spy1 > 0 and spy2 < 0:
        if side == "L":
            assert d.get("tlp", None) is None
            d["blp"] = indices[0]
            d["tlp"] = indices[1]
        elif side == "R":
            assert d.get("trp", None) is None
            d["brp"] = indices[0]
            d["trp"] = indices[1]
        else:
            assert False
    elif spy1 < 0 and spy2 > 0:
        if side == "L":
            assert d.get("tlp", None) is None
            d["blp"] = indices[0]
            d["tlp"] = indices[1]
        elif side == "R":
            assert d.get("trp", None) is None
            d["brp"] = indices[0]
            d["trp"] = indices[1]
        else:
            assert False
    else:
        d["errors"] = True


def handleCenterPoints(data, nbIndices, d):
    N = len(data)
    for i in range(N):
        y = data[i][1]
        if y < 0:
            d["bp"] = nbIndices[i]
        elif y == 0:
            d["cp"] = nbIndices[i]
        elif y > 0:
            d["tp"] = nbIndices[i]

    if d.get("bp", None) is None \
            or d.get("cp", None) is None \
            or d.get("bp", None) is None:
        d["errors"] = True


def triangleCombinations(d):
    combinations = list()
    combinationsTemp = [[d["tlp"], d["blp"], d["cp"]],
                        [d["trp"], d["brp"], d["cp"]],
                        [d["tp"], d["cp"], d["trp"]],
                        [d["tp"], d["cp"], d["tlp"]],
                        [d["bp"], d["cp"], d["brp"]],
                        [d["bp"], d["cp"], d["blp"]]]
    for comb in combinationsTemp:
        comb = np.array(comb)
        comb.sort()
        combinations.append(np.array(comb))
    return combinations


def removeDuplicateTriangles(triangles):
    triangles = set(map(tuple, triangles))
    triangles = np.array(list(set(triangles)), dtype=int)
    return triangles


def findTriangles(data, k=7, debug=False, useIntermediateResults=False):
    # TODO
    #  1. are the correct points chosen for the triangles?
    #  2. why are there more interpolated points than original points?
    #   are not all duplicates removed (?)
    #   lower bound: each hexagon gives 6 unique points,
    #   upper bound: a naive algorithm however will count each point thrice given sufficient hexagons
    #  3. does the information make sense?
    dists, indices = computeNearestNeighboursMatrix(data, k=k)

    N = len(data)
    triangles = list()
    errorIndices = list()
    for i in range(N):
        nbIndices = indices[i]
        alignedData = alignData(data[nbIndices])

        flps, lps, cps, rps, frps = determinePosition(alignedData, np.arange(k))
        if len(lps) == 2 and len(cps) == 3 and len(rps) == 2:
            # we have a hexagon structure

            d = dict()
            handlePointsToTheSide(alignedData[lps], nbIndices[lps], d, side="L")
            handlePointsToTheSide(alignedData[rps], nbIndices[rps], d, side="R")
            handleCenterPoints(alignedData[cps], nbIndices[cps], d)

            if d.get("errors") is None:
                triCombs = triangleCombinations(d)
                triangles = triangles + triCombs
            else:
                errorIndices.append(i)

    triangles = removeDuplicateTriangles(triangles)
    return triangles, errorIndices


def interpolateViaTriangles(voxelData, displacementData, k=7, debug=False, useIntermediateResults=False):
    # TODO
    #  1. is the correct information written out?
    #  2. does the information make sense?
    triangles, errorIndices = findTriangles(voxelData, k=k, debug=debug,
                                            useIntermediateResults=useIntermediateResults)

    df = pd.DataFrame(columns=interpolatedDataHeaders.allHeaders)
    N = len(triangles)
    for i in range(N):
        d = dict()
        tri = np.array(triangles[i], dtype=int)
        vd = voxelData[tri]
        dd = displacementData[tri]

        mvd = np.mean(vd, axis=0)
        mdd = np.mean(dd, axis=0)

        totalDisp = np.linalg.norm(mdd, ord=2)

        d[interpolatedDataHeaders.x] = mvd[0]
        d[interpolatedDataHeaders.y] = mvd[1]
        d[interpolatedDataHeaders.z] = mvd[2]
        d[interpolatedDataHeaders.xDisplacement] = mdd[0]
        d[interpolatedDataHeaders.yDisplacement] = mdd[1]
        d[interpolatedDataHeaders.zDisplacement] = mdd[2]
        d[interpolatedDataHeaders.totalDisplacement] = totalDisp

        df = df.append(d.copy(), ignore_index=True)

    return df
