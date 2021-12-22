import time


class metadataDsts:
    scPlot = "scatterplots"
    pr = "projections"
    hg = "histograms"
    spc = "samplesPerCluster"
    nnCount = "nearestNeighboursCountsCsv"


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from backbone import computeNearestNeighboursMatrix, alignData
from helpers.csvTools import csv_to_xlsx, listsToCsv
from helpers.general_tools import getColours, sciF, execFunctionOnMatrix, removeDir
from helpers.seabornTools import createScatterPlot, plotHistogram, nearestNeighbourscatterPlot, scatterPlot3D


def createScattterPlots(data, dstDir=None, show=False, save=True, dpi=500):
    dstDir = dstDir + os.sep + "scatterPlots"
    dd2d = "dreieckData2d"
    dd3d = "dreieckData3d"
    vd2d = "viereckData2d"
    vd3d = "viereckData3d"

    createScatterPlot(data[:, :2], dst=dstDir, title=dd2d, save=save, show=show, dpi=dpi)
    createScatterPlot(data, title=dd3d, dst=dstDir, save=save, show=show, dpi=dpi)
    # createScatterPlot(viereckData[:, :2], title=vd2d, save=save, show=show, dpi=dpi)
    # createScatterPlot(viereckData, title=vd3d, save=save, show=show, dpi=dpi)


def createAxisProjections(data, dstDir=None, show=False, save=True, dpi=500):
    xDreieckData = data[:, 0]
    yDreieckData = data[:, 1]

    ddpx = "dreieckDataProjectionX"
    ddpy = "dreieckDataProjectionY"
    vdpx = "viereckDataProjectionX"
    vdpy = "viereckDataProjectionY"

    dst = dstDir + os.sep + "axisProjections"

    createScatterPlot(xDreieckData, dst=dst, title=ddpx, save=save, show=show, dpi=dpi)
    createScatterPlot(yDreieckData, dst=dst, title=ddpy, save=save, show=show, dpi=dpi)
    # createScatterPlot(xViereckData, dst=base, title=vdpx, save=save, show=show, dpi=dpi)
    # createScatterPlot(yViereckData, dst=base, title=vdpy, save=save, show=show, dpi=dpi)


def findBestKForKMeans2(data):
    print("Starting up: ", "findBestKForKMeans2")
    data = data[:, :1]

    fitted_kmeans = {}
    labels_kmeans = {}
    df_scores = []
    k_values_to_try = np.arange(2, 40)
    for n_clusters in k_values_to_try:
        # Perform clustering.
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=0, max_iter=50, n_init=100)
        labels_clusters = kmeans.fit_predict(data)

        # Insert fitted model and calculated cluster labels in dictionaries,
        # for further reference.
        fitted_kmeans[n_clusters] = kmeans
        labels_kmeans[n_clusters] = labels_clusters

        # Calculate various scores, and save them for further reference.
        silhouette = silhouette_score(data, labels_clusters)
        ch = calinski_harabasz_score(data, labels_clusters)
        db = davies_bouldin_score(data, labels_clusters)
        tmp_scores = {"n_clusters": n_clusters,
                      "silhouette_score": silhouette,
                      "calinski_harabasz_score": ch,
                      "davies_bouldin_score": db,
                      }
        df_scores.append(tmp_scores)

    # Create a DataFrame of clustering scores, using `n_clusters` as index, for easier plotting.
    df_scores = pd.DataFrame(df_scores)
    df_scores.set_index("n_clusters", inplace=True)

    dst = metadataDsts.spc
    df_scores.to_csv("n_clusters.csv", sep=";")
    csv_to_xlsx("n_clusters.csv")
    print("Done with: ", findBestKForKMeans2)


def findBestKForKMeans(data, dstDir=None, show=False, save=False):
    print("Starting up: ", "findBestKForKMeans")
    dstDir = dstDir + os.sep + "bestKForKMeans"
    os.makedirs(dstDir, exist_ok=True)

    X = np.copy(data[:, :2])
    X[:, 1] = 0
    wcss = []
    N = np.min([len(data) - 1, 30])
    for i in range(2, N):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=50, n_init=100, random_state=0)
        pred_y = kmeans.fit_predict(data)
        fig = plt.figure()
        ax = plt.axes()

        plt.plot(data[:, 0], data[:, 1], ".", alpha=0.03)
        if save:
            ffp = dstDir + os.sep + "clusters_" + str(i)
            plt.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], ".")
            plt.savefig(ffp, dpi=500)
        plt.close(fig)

    print("Done with: ", "findBestKForKMeans")


def computeNormalizedNearestNeighboursMatrixTest(data, dstDir=None, show=False, save=False):
    X = np.copy(data)
    dstDir = dstDir + os.sep + "nearestNeighbourMatrices"

    Ms = np.max(X, axis=0)
    X = X / Ms

    (dists, indices) = computeNearestNeighboursMatrix(X)

    uniques, counts = np.unique(indices, return_counts=True)
    uniques2, counts2 = np.unique(counts, return_counts=True)

    uniques = ["Uniques"] + uniques.tolist()
    counts = ["Counts"] + counts.tolist()
    dst = dstDir + os.sep + "countsWithNormalization.csv"
    countsCsvFp = listsToCsv([uniques, counts], dst=dst)
    csv_to_xlsx(countsCsvFp)

    uniques2 = ["Uniques"] + uniques2.tolist()
    counts2 = ["Counts"] + counts2.tolist()
    dst = dstDir + os.sep + "countsWithNormalization2.csv"
    countsCsvFp2 = listsToCsv([uniques2, counts2], dst=dst)
    csv_to_xlsx(countsCsvFp2)


def computeNotNormalizedNearestNeighboursMatrixTest(data, dstDir=None, show=False, save=False):
    X = np.copy(data)
    dstDir = dstDir + os.sep + "nearestNeighbourMatrices"

    (dists, indices) = computeNearestNeighboursMatrix(X)

    uniques, counts = np.unique(indices, return_counts=True)
    uniques2, counts2 = np.unique(counts, return_counts=True)

    uniques = ["Uniques"] + uniques.tolist()
    counts = ["Counts"] + counts.tolist()
    uniques2 = ["Uniques"] + uniques2.tolist()
    counts2 = ["Counts"] + counts2.tolist()

    countsCsv = "countsWithoutNormalization.csv"
    dst = dstDir + os.sep + countsCsv
    countsCsvFp = listsToCsv([uniques, counts], dst=dst)
    csv_to_xlsx(src=countsCsvFp)

    countsCsv2 = "countsWithoutNormalization2.csv"
    dst = dstDir + os.sep + countsCsv2
    countsCsvFp2 = listsToCsv([uniques2, counts2], dst=dst)
    csv_to_xlsx(src=countsCsvFp2)


def computeCountsTest(data, dstDir=None, show=False, save=False):
    X = np.copy(data)
    dstDir = dstDir + os.sep + "nearestNeighboursCountsAsCsv"
    os.makedirs(dstDir, exist_ok=True)

    (dists, indices) = computeNearestNeighboursMatrix(X)

    uniques, counts = np.unique(indices, return_counts=True)
    uniques2, counts2 = np.unique(counts, return_counts=True)

    Ms = np.max(np.abs(X), axis=0)
    X = X / Ms

    (dists2, indices2) = computeNearestNeighboursMatrix(X)

    uniques3, counts3 = np.unique(indices2, return_counts=True)
    uniques4, counts4 = np.unique(counts3, return_counts=True)

    uniquesWith = ["UniquesWithNormalization"] + uniques.tolist()
    countsWith = ["CountsWithNormalization"] + counts.tolist()
    countsWithout = ["CountsWithoutNormalization"] + counts3.tolist()

    countsCsv = "counts.csv"
    dst = dstDir + os.sep + countsCsv
    countsCsvFp = listsToCsv([uniquesWith, countsWith, countsWithout], dst=dst, withDate=False)
    csv_to_xlsx(countsCsvFp)

    uniquesWith = ["UniquesWithNormalization"] + uniques2.tolist()
    countsWith = ["CountsWithNormalization"] + counts2.tolist()
    countsWithout = ["CountsWithoutNormalization"] + counts4.tolist()

    countsCsv = "counts2.csv"
    dst = dstDir + os.sep + countsCsv
    countsCsvFp = listsToCsv([uniquesWith, countsWith, countsWithout], dst=dst, withDate=False)
    csv_to_xlsx(countsCsvFp)


def countsByCluster(data, dstDir=None, show=False, save=False):
    print("Starting up: ", "countsByCluster")

    X = np.copy(data[:, :2])
    X[:, 1] = 0

    dstDir = dstDir + os.sep + "countsByCluster"
    os.makedirs(dstDir, exist_ok=True)
    N = np.min([len(X) - 1, 40])
    for i in range(2, N):
        name = "clusters_" + str(i)

        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=50, n_init=100, random_state=0)
        pred_y = kmeans.fit_predict(X)
        counts = np.unique(pred_y, return_counts=True)

        clusterName = ["cluster"] + list(counts[0])
        occurence = ["count"] + list(counts[1])

        if save:
            dst = dstDir + os.sep + name
            ffp = listsToCsv([clusterName, occurence], dst=dst, withDate=False)
            csv_to_xlsx(ffp)
    print("Done with: ", "countsByCluster")


def elbowMethod(data, dstDir=None, show=False, save=True):
    print("Starting up: ", elbowMethod)

    X = np.copy(data[:, :2])
    X[:, 1] = 0
    print(data)
    sse = []
    for k in range(2, 40):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=100, random_state=0)
        kmeans.fit(data)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(X)
        curr_sse = 0
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(X)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (data[i, 0] - curr_center[0]) ** 2
        sse.append(curr_sse)

    for ss in sse:
        print(ss)
    plt.plot(np.arange(2, 40), sse)
    if show:
        plt.show()
    plt.close("all")
    print("Done with: ", "elbowMethod")


def createProjectedDataHistograms(data, dstDir=None, show=False, save=True, dpi=500):
    X = np.copy(data[:, 0])
    dstDir = dstDir + os.sep + "projectedDataHistograms"
    for i in range(2, 40):
        title = "bins_" + str(i)
        if save:
            plotHistogram(data=X, title=title, save=save, show=show, dpi=dpi, bins=i,
                          dst=dstDir)  #
    plt.close("all")
    print("Done with: ", "createHistograms")


def visualizeCounts(data, dstDir=None, show=False, save=False):
    ##### part1
    X = np.copy(data)
    (dists, indices) = computeNearestNeighboursMatrix(X)

    uniques, counts = np.unique(indices, return_counts=True)
    uniques2, counts2 = np.unique(counts, return_counts=True)

    N = len(uniques)
    M = len(uniques2)
    colourSpace = getColours(M)
    colours = np.zeros((N, 4))

    for i in range(N):
        count = counts[i]
        cIndex = np.where(uniques2 == count)[0]
        colours[i, :] = colourSpace[cIndex]

    fn = "visualCountsWithoutNormalization"
    dst = dstDir + os.sep + "visualCounts"

    nearestNeighbourscatterPlot(X, title=fn, dst=dst,
                                save=save, show=show,
                                dpi=500, colours=colours, labels=counts, alpha=0.5)
    ##### part2
    X = np.copy(data)
    Ms = np.max(np.abs(X), axis=0)
    X = X / Ms
    X[:, :] = X[:, :]
    (dists, indices) = computeNearestNeighboursMatrix(X)

    uniques, counts = np.unique(indices, return_counts=True)
    uniques2, counts2 = np.unique(counts, return_counts=True)

    N = len(uniques)
    M = len(uniques2)
    colourSpace = getColours(M)
    colours = np.zeros((N, 4))

    for i in range(N):
        count = counts[i]
        cIndex = np.where(uniques2 == count)[0]
        colours[i, :] = colourSpace[cIndex]

    fn = "visualCountsWithNormalization"
    nearestNeighbourscatterPlot(X, title=fn, dst=dst,
                                save=save, show=show,
                                dpi=500, colours=colours, labels=counts, alpha=0.5)
    plt.close("all")


def saveDistances(data, dstDir, save=False):
    X = np.copy(data)
    nnn = 12
    (dists, indices) = computeNearestNeighboursMatrix(X, nnn + 1)
    dists = dists[..., 1:]
    args = dict()
    args["k"] = 5
    dists = execFunctionOnMatrix(dists, sciF, args)
    fn = "distances.csv"
    dstDir = dstDir + os.sep + "distance"
    os.makedirs(dstDir, exist_ok=True)

    headers = ["id"] + ["dist_" + str(i) for i in range(1, nnn + 2)]

    N = len(X)
    ids = np.arange(N).tolist()
    data = list()
    content = None
    for i in range(nnn + 1):
        content = list()
        content.append(headers[i])

        if i == 0:
            content = content + ids
        else:
            col = dists[:, i - 1].tolist()
            content = content + col
        data.append(content)

    ffp = listsToCsv(data, dst=dstDir + os.sep + fn, debug=True, deleteIfExists=True)
    csv_to_xlsx(ffp, ffp)


def randomHexagonSampling(data, dstDir, withCenter=True):
    X = np.copy(data)
    nnn = 6
    (dists, indices) = computeNearestNeighboursMatrix(X, nnn + 1)

    if not withCenter:
        indices = indices[..., 1:]
    dstDir = dstDir + os.sep + "hexSampling"
    if os.path.exists(dstDir):
        removeDir(dstDir)
        time.sleep(1)
    os.makedirs(dstDir, exist_ok=True)

    N = len(data)
    for i in range(100):
        rngSamIndex = np.random.randint(N)
        rngSamIndices = indices[rngSamIndex, :]
        chosenSamples = data[rngSamIndices]

        title = str(rngSamIndex) + ".png"
        createScatterPlot(data=chosenSamples, title=title, dst=dstDir,
                          save=True, show=False, dpi=500, colours=None,
                          labels=None, alpha=1.0)


def detailedRandomHexagonSampling(data, dstDir, numSamples=100, withCenter=True):
    X = np.copy(data)
    nnn = 6
    (dists, indices) = computeNearestNeighboursMatrix(X, nnn + 1)

    if not withCenter:
        indices = indices[..., 1:]
    dstDir = dstDir + os.sep + "detailedHexSampling"
    if os.path.exists(dstDir):
        removeDir(dstDir)
        time.sleep(1)
    os.makedirs(dstDir, exist_ok=True)

    N = len(data)

    headers = "centerPoindId;" + "neighbourIds;" + "centerCords;dists;"
    for i in range(6):
        headers = headers + "sampleDist_" + str(i) + ";" + \
                  "sampleCoords" + str(i) + ";" + \
                  "sample" + str(i) + "_x;" + \
                  "sample" + str(i) + "_y;" + \
                  "sample" + str(i) + "_z;"

    headers = headers[:-1]
    rngs = np.arange(N)
    np.random.shuffle(rngs)

    f = open(dstDir + os.sep + "detailedHexSampling.csv", "a+")
    f.write(headers + "\n")
    # dists = execFunctionOnMatrix(dists, sciF, {"k": 5})
    for i in range(numSamples):
        rngSamIndex = rngs[i]
        rngSamIndices = indices[rngSamIndex, :]
        chosenSamples = data[rngSamIndices]

        rngSamDists = dists[rngSamIndex, :]

        title = str(rngSamIndex) + ".png"
        createScatterPlot(data=chosenSamples, title=title, dst=dstDir,
                          save=True, show=False, dpi=500, colours=None,
                          labels=None, alpha=1.0)
        s = str(rngSamIndex) + ";" + \
            str(rngSamIndices[1:]) + ";" + \
            str(data[rngSamIndex]) + ";" + \
            str(rngSamDists[1:]) + ";"
        centerPoint = data[rngSamIndices[0]]
        sampledPoints = data[rngSamIndices[1:]]
        for j in range(len(sampledPoints)):
            p = sampledPoints[j]
            diff = centerPoint - p
            s = s + str(rngSamDists[1 + j]) + ";"
            s = s + str(p) + ";"
            s = s + str(diff[0]) + ";"
            s = s + str(diff[1]) + ";"
            s = s + str(diff[2]) + ";"
        s = s[:-1]
        f.write(s + "\n")

    f.close()


def viewTransformedDataSurroundings(data, dstDir, nnn=6):
    N = len(data)
    dstDir = dstDir + os.sep + "viewLocalShape"
    if os.path.exists(dstDir):
        removeDir(dstDir)
        time.sleep(1)
    os.makedirs(dstDir, exist_ok=True)

    nnn = 6
    (dists, indices) = computeNearestNeighboursMatrix(data, nnn + 1)
    for i in range(N):
        newData = alignData(data[indices[i]])

        newData = newData[:, :2]
        newData = newData - newData[0]
        title = str(i) + ".png"
        createScatterPlot(data=newData, title=title, dst=dstDir,
                          save=True, show=False, dpi=500, colours=None,
                          labels=None, alpha=1.0)


def isParallelogram(l, c, r):
    if l == 3 and c == 4:
        return True
    elif l == 4 and c == 3:
        return True
    elif r == 3 and c == 4:
        return True
    elif r == 4 and c == 3:
        return True
    elif r == 3 and c == 3 and l == 1:
        return True
    elif l == 3 and c == 3 and r == 1:
        return True
    else:
        return False


def isTrapez(l, c, r):
    if l == 3 and c == 2 and r == 2:
        return True
    elif r == 3 and c == 2 and l == 2:
        return True
    else:
        return False


def veryWeird(farleft, leftCount, centerCount, rightCount, farright):
    if farleft != 0 and farright != 0:
        return True
    elif farleft == 0 and farright == 0 and (leftCount == 4 or rightCount == 4):
        return True
    else:
        return False


def countHexagones(data, dstDir, nnn=6):
    N = len(data)
    dstDir = dstDir + os.sep + "localShape"
    if os.path.exists(dstDir):
        removeDir(dstDir)
        time.sleep(1)
    os.makedirs(dstDir, exist_ok=True)

    (dists, indices) = computeNearestNeighboursMatrix(data, nnn + 1)

    headers = "index;farleft;left;center;right;farright;type"

    destFile = dstDir + os.sep + "localShape.csv"
    f = open(destFile, "a+")
    f.write(headers + "\n")
    for i in range(N):
        newData = alignData(data[indices[i]])

        newData = newData[:, :2]
        newData = newData - newData[0]

        farleft = 0
        leftCount = 0
        centerCount = 0
        rightCount = 0
        farright = 0

        cP = newData[0]
        t = "-1"

        "[-1, -0.4, 0, 0.4, 1]"

        for p in newData:
            if p[0] > -0.25 and p[0] < 0.25:
                centerCount = centerCount + 1
            elif p[0] > 0.25 and p[0] < 0.75:
                rightCount = rightCount + 1
            elif p[0] < -0.25 and p[0] > -0.75:
                leftCount = leftCount + 1
            elif p[0] > 0.75:
                farright = farright + 1
            elif p[0] < -0.75:
                farleft = farleft + 1
            else:
                assert False

        if centerCount == 3 and leftCount == 2 and rightCount == 2:
            t = "hexagon"
        elif isParallelogram(leftCount, centerCount, rightCount):
            t = "parallelogram"
        elif isTrapez(leftCount, centerCount, rightCount):
            t = "trapez"
        elif (farleft == 0 and farright != 0) or (farright == 0 and farleft != 0):
            t = "weird"
        elif veryWeird(farleft, leftCount, centerCount, rightCount, farright):
            t = "very weird"
        else:
            print(i, farleft, leftCount, centerCount, rightCount, farright)
            f.close()
            assert False

        s = ""
        s = str(i) + ";" + str(farleft) + ";" + str(leftCount) + ";" + str(centerCount) + \
            ";" + str(rightCount) + ";" + str(farright) + ";" + t
        f.write(s + "\n")
    f.close()


def colourDataByShape(data, dstDir, nnn=6):
    N = len(data)
    dstDir = dstDir + os.sep + "colourDataByShape"
    if os.path.exists(dstDir):
        removeDir(dstDir)
        time.sleep(1)
    os.makedirs(dstDir, exist_ok=True)

    (dists, indices) = computeNearestNeighboursMatrix(data, nnn + 1)

    colours = list()

    for i in range(N):
        newData = alignData(data[indices[i]])

        newData = newData[:, :2]
        newData = newData - newData[0]

        farleft = 0
        leftCount = 0
        centerCount = 0
        rightCount = 0
        farright = 0

        cP = newData[0]
        t = "-1"

        "[-1, -0.4, 0, 0.4, 1]"

        for p in newData:
            if p[0] > -0.25 and p[0] < 0.25:
                centerCount = centerCount + 1
            elif p[0] > 0.25 and p[0] < 0.75:
                rightCount = rightCount + 1
            elif p[0] < -0.25 and p[0] > -0.75:
                leftCount = leftCount + 1
            elif p[0] > 0.75:
                farright = farright + 1
            elif p[0] < -0.75:
                farleft = farleft + 1
            else:
                assert False

        if centerCount == 3 and leftCount == 2 and rightCount == 2:
            t = "hexagon"
            colours.append("green")
        elif isParallelogram(leftCount, centerCount, rightCount):
            t = "parallelogram"
            colours.append("yellow")
        elif isTrapez(leftCount, centerCount, rightCount):
            t = "trapez"
            colours.append("yellow")
        elif (farleft == 0 and farright != 0) or (farright == 0 and farleft != 0):
            t = "weird"
            colours.append("red")
        elif veryWeird(farleft, leftCount, centerCount, rightCount, farright):
            t = "very weird"
            colours.append("red")
        else:
            print(i, farleft, leftCount, centerCount, rightCount, farright)
            assert False
    colours = np.array(colours)
    scatterPlot3D(data, dst=dstDir, title="visualization1.png", colours=colours, alpha=0.5)
    scatterPlot3D(data, dst=dstDir, title="visualization2.png", colours=colours, alpha=0.25)
    scatterPlot3D(data, dst=dstDir, title="visualization3.png", colours=colours, alpha=0.03)