import os

from data.dataKeys import viereckRasterHeaders, dreieckRasterHeaders
from helpers.csvTools import loadDataframe, readCsv
from helpers.general_tools import recursiveWalk
from paths import projectBase, DataDir
import numpy as np


class AramisVersions:
    V6 = "AramisV6"
    V2020 = "Aramis2020"


def loadAllDataAsDfs(src, N=100):
    files = os.listdir(src)
    dfs = list()

    for f in files:
        ffp = src + os.sep + f

        if ffp.endswith(".csv"):
            df = readCsv(ffp)
            dfs.append(df)
        if N is not None:
            if len(dfs) == N:
                break
    return dfs


def loadDataByPath(src):
    ffp = src
    df = None
    if ffp.endswith(".csv"):
        df = readCsv(ffp)
    else:
        df = readCsv(ffp + ".csv")
    return df


def loadAllDataAsArrays(src, normalizeData=False, N=None):
    dfs = loadAllDataAsDfs(src=src)
    arrs = list()
    for df in dfs:
        arr = retrieveVoxels(df, normalizeData)
        arrs.append(arr)
        if N is not None:
            if len(arrs) == N:
                break
    return arrs


def dfToDictOfArrays(df, selectedHeaders=None):
    keys = None
    d = dict()
    if selectedHeaders is None:
        selectedHeaders = df.keys()

    for sc in selectedHeaders:
        col = df[sc].tolist()
        d[sc] = col
    return d


def loadSingleFrame(src, aramisVersion="AramisV6"):
    if aramisVersion == AramisVersions.V2020:
        df = loadDataframe(src)


def retrieveVoxels(df, normalizeData=False, aramisVersion=AramisVersions.V6):
    if aramisVersion == AramisVersions.V6:
        x = df[dreieckRasterHeaders.x].tolist()
        y = df[dreieckRasterHeaders.y].tolist()
        z = df[dreieckRasterHeaders.z].tolist()
    elif aramisVersion == AramisVersions.V2020:
        x = df[viereckRasterHeaders.x_Koordinate].tolist()
        y = df[viereckRasterHeaders.y_Koordinate].tolist()
        z = df[viereckRasterHeaders.z_Koordinate].tolist()
        q = [x, y, z]
        N = len(x)
        for i in range(N):
            for j in range(3):
                e: str = q[j][i]
                for k in [5, 6, 7, 8, 9]:
                    if k + 1 == len(e):
                        break
                    if e[k] == ".":
                        e = e[:k] + e[k + 1]
                        break
                q[j][i] = e

    N = len(df)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    if normalizeData:
        x = x / np.max(np.abs(x))
        y = y / np.max(np.abs(y))
        z = z / np.max(np.abs(z))

    data = np.zeros((N, 3))
    data[:, 0] = x
    data[:, 1] = y
    data[:, 2] = z

    return data


def retrieveTotalDisplacement(df, normalizeData=False):
    x = df[dreieckRasterHeaders.displacement_x].tolist()
    y = df[dreieckRasterHeaders.displacement_y].tolist()
    z = df[dreieckRasterHeaders.displacement_z].tolist()

    N = len(df)

    x = np.abs(np.array(x))
    y = np.abs(np.array(y))
    z = np.abs(np.array(z))
    disp = np.concatenate([x, y, z], axis=1)
    totalDisplacement = np.linalg.norm(disp, axis=1, ord=2)  # x + y + z  # does l2 norm make a big difference(?)
    if normalizeData:
        totalDisplacement = totalDisplacement / np.max(totalDisplacement)
    return totalDisplacement


def retrieveDisplacements(df):
    x = df[dreieckRasterHeaders.displacement_x].tolist()
    y = df[dreieckRasterHeaders.displacement_y].tolist()
    z = df[dreieckRasterHeaders.displacement_z].tolist()

    x = np.array(x)[:, np.newaxis]
    y = np.array(y)[:, np.newaxis]
    z = np.array(z)[:, np.newaxis]
    displacements = np.concatenate([x, y, z], axis=1)
    return displacements


def getAllPathes(sd=".", filterFuntion=None, debug=False):
    ds, fs = recursiveWalk(sd, filterFuntion, debug=debug)
    return ds, fs
